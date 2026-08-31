"""Billable compatibility checks for boundaries that deterministic fakes cannot prove."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from repo_rivet.approval.llm_reviewer import OpenAIApprovalReviewer
from repo_rivet.approval.models import (
    AnalysisLevel,
    ApprovalFacts,
    ApprovalRequest,
    Capability,
    EffectScope,
    ExecutableOrigin,
    OperationClass,
    RiskAssessment,
    RiskLevel,
)
from repo_rivet.config import ApiConfig
from repo_rivet.llm.base import ModelRequestOptions
from repo_rivet.llm.openai_compatible import OpenAICompatibleClient
from repo_rivet.memory.models import MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.planning.classifier import OpenAIPlanClassifier, WorkspacePlanningSummary
from repo_rivet.subagents.manager import SubagentManager
from repo_rivet.subagents.models import DelegateTaskArguments, SubagentProfile
from repo_rivet.tools.base import ToolResult

pytestmark = pytest.mark.live_api


class RecordingEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log(self, event_type: str, **data: Any) -> None:
        self.events.append((event_type, data))


def test_live_streaming_tool_call_and_follow_up_protocol(live_api_config: ApiConfig) -> None:
    """Exercise streaming JSON assembly and provider reasoning-state replay together."""
    events = RecordingEvents()
    client = OpenAICompatibleClient(live_api_config, event_logger=events)
    tool = {
        "type": "function",
        "function": {
            "name": "report_probe",
            "description": "Return the exact compatibility probe token.",
            "parameters": {
                "type": "object",
                "properties": {"token": {"type": "string"}},
                "required": ["token"],
                "additionalProperties": False,
            },
        },
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Call report_probe exactly once with token REPORIVET_LIVE_TOOL_OK. "
                "Do not answer in text yet."
            ),
        }
    ]
    first = client.complete(
        messages=messages,
        tools=[tool],
        options=ModelRequestOptions(required_tool="report_probe"),
    )

    assert len(first.tool_calls) == 1
    call = first.tool_calls[0]
    assert call.name == "report_probe"
    assert call.arguments == {"token": "REPORIVET_LIVE_TOOL_OK"}
    messages.extend(
        [
            first.as_assistant_message(),
            ToolResult(ok=True, output="probe accepted").as_tool_message(call.id),
            {
                "role": "user",
                "content": "Now reply with exactly REPORIVET_LIVE_FOLLOWUP_OK and no tool call.",
            },
        ]
    )
    second = client.complete(messages=messages, tools=[tool])

    assert not second.tool_calls
    assert second.content is not None
    assert second.content.strip() == "REPORIVET_LIVE_FOLLOWUP_OK"
    assert any(name == "model_stream_progress" for name, _data in events.events)
    if first.input_tokens is not None:
        assert first.input_tokens > 0
    if second.output_tokens is not None:
        assert second.output_tokens > 0


def test_live_auxiliary_classifiers_return_valid_contracts(
    live_api_config: ApiConfig,
    tmp_path: Path,
) -> None:
    """Cover the separate non-streaming planning and approval API paths."""
    classifier = OpenAIPlanClassifier(live_api_config, timeout_seconds=90)
    classification = classifier.classify(
        "Fix the spelling of one known word in README.txt and change no other file.",
        WorkspacePlanningSummary(
            empty=False,
            sampled_files=1,
            sampled_directories=0,
            truncated=False,
            extensions=(".txt",),
            root_entries=("README.txt",),
        ),
    )
    assert classification is not None
    assert classification.decision in {"plan", "execute"}

    reviewer = OpenAIApprovalReviewer(live_api_config, timeout_seconds=90)
    explicit_effects = {
        "compile_workspace_code",
        "filesystem_read",
        "filesystem_write",
        "process_execution",
    }
    review = reviewer.review(
        ApprovalRequest(
            request_id="live-review",
            session_id="live-session",
            tool_name="run_command",
            arguments={"command": "g++ -o app app.cpp"},
            normalized_arguments={
                "command": {"program": "g++", "args": ["-o", "app", "app.cpp"]},
                "timeout_seconds": 60,
                "_resolved_paths": {"cwd": str(tmp_path)},
            },
            declared_capabilities={Capability.PROCESS_EXECUTE},
            workspace=str(tmp_path),
            fingerprint="live-review-fingerprint",
            assessment=RiskAssessment(level=RiskLevel.MEDIUM),
            task_summary="Build the requested C++ program inside the workspace.",
            facts=ApprovalFacts(
                operation_class=OperationClass.BUILD,
                analysis_level=AnalysisLevel.EXACT,
                executable="g++",
                resolved_executable="/usr/bin/g++",
                executable_origin=ExecutableOrigin.TRUSTED_TOOLCHAIN,
                read_paths=["app.cpp"],
                write_paths=["app"],
                effect_scope=EffectScope.WORKSPACE,
                explicit_effects=explicit_effects,
                constraints={"shell_free_argv", "workspace_cwd", "timeout_enforced"},
                task_relevance="required",
            ),
        )
    )
    assert review is not None, reviewer.last_failure
    assert explicit_effects.issubset(review.recognized_effects)


def test_live_read_only_subagent_returns_validated_report(
    live_api_config: ApiConfig,
    tmp_path: Path,
) -> None:
    """Run the real provider through the isolated child Controller and report validator."""
    workspace = tmp_path / "workspace"
    source = workspace / "src"
    source.mkdir(parents=True)
    (source / "app.py").write_text(
        "def answer() -> int:\n    return 42\n",
        encoding="utf-8",
    )
    parent_store = MemoryStore(tmp_path / "parent-session")
    parent_memory = MemoryState(session_id="live-parent", workspace_revision=0)
    events = RecordingEvents()
    manager = SubagentManager(
        workspace=workspace,
        parent_store=parent_store,
        model_client_factory=lambda child_events: OpenAICompatibleClient(
            live_api_config,
            event_logger=child_events,
        ),
        event_logger=events,
        max_concurrency=2,
    )
    manager.bind(parent_memory)

    result = manager.delegate(
        DelegateTaskArguments(
            profile=SubagentProfile.EXPLORER,
            objective="Locate the function that returns the numeric answer in src/app.py.",
            deliverable="Report the function name, return value, file path, and observed evidence.",
            scope_paths=["src"],
            constraints=["Read only src/app.py", "Keep the report concise"],
        )
    )

    assert result.ok, result.error
    payload = json.loads(result.output)
    assert payload["profile"] == "explorer"
    assert payload["status"] == "completed"
    assert payload["freshness"] == "fresh"
    assert payload["findings"]
    assert parent_memory.modified_files == set()
    assert any(name == "subagent_report_accepted" for name, _data in events.events)
