import json
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from repo_rivet.approval.engine import ApprovalEngine
from repo_rivet.approval.grant_store import ApprovalGrantStore
from repo_rivet.approval.hard_policy import HardSafetyPolicy
from repo_rivet.approval.human_approver import (
    NonInteractiveHumanApprover,
    TerminalHumanApprover,
)
from repo_rivet.approval.llm_reviewer import OpenAIApprovalReviewer
from repo_rivet.approval.models import (
    AnalysisLevel,
    ApprovalAction,
    ApprovalDecision,
    ApprovalMode,
    ApprovalRequest,
    ApprovalScope,
    ArtifactProvenance,
    Capability,
    ExecutableOrigin,
    LLMReviewResult,
    OperationClass,
    RiskLevel,
)
from repo_rivet.approval.normalizer import RequestNormalizer
from repo_rivet.approval.review_context import build_review_payload
from repo_rivet.approval.risk_analyzer import RiskAnalyzer
from repo_rivet.approval.semantic_analyzer import ApprovalFactAnalyzer
from repo_rivet.config import ApiConfig
from repo_rivet.memory.models import MemoryState
from repo_rivet.tools.base import ToolCall
from repo_rivet.tools.registry import create_default_registry
from repo_rivet.verification.models import (
    CommandSpec,
    VerificationCheck,
    VerificationKind,
    VerificationPlan,
)


class FakeHumanApprover:
    def __init__(
        self,
        *,
        action: ApprovalAction = ApprovalAction.ALLOW,
        scope: ApprovalScope = ApprovalScope.ONCE,
        guidance: str | None = None,
    ) -> None:
        self.action = action
        self.scope = scope
        self.guidance = guidance
        self.requests: list[ApprovalRequest] = []
        self.llm_reviews: list[LLMReviewResult | None] = []

    def ask(
        self,
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None = None,
    ) -> ApprovalDecision:
        self.requests.append(request)
        self.llm_reviews.append(llm_review)
        return ApprovalDecision(
            action=self.action,
            source="human",
            reason="test decision",
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
            scope=self.scope,
            guidance=self.guidance,
        )


class FakeReviewer:
    def __init__(self, result: LLMReviewResult | None) -> None:
        self.result = result
        self.requests: list[ApprovalRequest] = []

    def review(self, request: ApprovalRequest) -> LLMReviewResult | None:
        self.requests.append(request)
        return self.result


class TimeoutReviewer:
    def review(self, request: ApprovalRequest) -> LLMReviewResult | None:
        raise TimeoutError("review timed out")


class EventCollector:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def log(self, event_type: str, **data: object) -> None:
        self.events.append((event_type, data))


def create_engine(
    workspace: Path,
    *,
    mode: ApprovalMode,
    human: FakeHumanApprover | NonInteractiveHumanApprover | None = None,
    reviewer: FakeReviewer | None = None,
) -> tuple[ApprovalEngine, MemoryState]:
    memory = MemoryState(session_id="approval-test")
    engine = ApprovalEngine(
        mode=mode,
        normalizer=RequestNormalizer(workspace),
        risk_analyzer=RiskAnalyzer(),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=human or FakeHumanApprover(),
        llm_reviewer=reviewer,
    )
    return engine, memory


def test_allow_all_approves_workspace_write_but_denies_escape(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    registry = create_default_registry(tmp_path, approval_engine=engine)

    allowed = registry.execute(
        ToolCall(
            id="write-1",
            name="write_file",
            arguments={"path": "inside.txt", "content": "ok"},
        )
    )
    denied = registry.execute(
        ToolCall(
            id="write-2",
            name="write_file",
            arguments={"path": "../outside.txt", "content": "no"},
        )
    )

    assert allowed.ok
    assert (tmp_path / "inside.txt").read_text(encoding="utf-8") == "ok"
    assert not denied.ok
    assert denied.error_code == "hard_policy_denied"
    assert not (tmp_path.parent / "outside.txt").exists()


def test_safe_auto_allows_normal_read_and_denies_sensitive_read(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    registry = create_default_registry(tmp_path, approval_engine=engine)

    normal = registry.execute(
        ToolCall(id="read-1", name="read_file", arguments={"path": "main.py"})
    )
    sensitive = registry.execute(
        ToolCall(id="read-2", name="read_file", arguments={"path": ".env"})
    )

    assert normal.ok
    assert not sensitive.ok
    assert sensitive.error_code == "hard_policy_denied"
    assert human.requests == []


def test_safe_auto_treats_git_status_as_typed_read(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    outcome = engine.authorize(
        tool_name="git_status",
        arguments={"path": "."},
        capabilities={Capability.FILESYSTEM_READ},
        session_id=engine.session_id,
    )

    assert outcome.decision.action == ApprovalAction.ALLOW
    assert outcome.decision.source == "safe_rule"
    assert outcome.request.facts.operation_class == OperationClass.READ
    assert human.requests == []


def test_edit_approval_contains_preflight_diff_and_version_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "main.py"
    path.write_text("value = 1\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    events = EventCollector()
    engine.event_logger = events
    registry = create_default_registry(
        tmp_path,
        approval_engine=engine,
        event_logger=events,
    )
    read = registry.execute(
        ToolCall(id="read-edit", name="read_file", arguments={"path": "main.py"})
    )
    assert read.ok and read.metadata

    edited = registry.execute(
        ToolCall(
            id="edit-1",
            name="edit_file",
            arguments={
                "path": "main.py",
                "snapshot_id": read.metadata["snapshot_id"],
                "operations": [
                    {
                        "op": "replace",
                        "start_line": 1,
                        "end_line": 1,
                        "new_lines": ["value = 2"],
                    }
                ],
            },
        )
    )

    assert edited.ok and edited.metadata
    request = human.requests[-1]
    normalized = request.normalized_arguments
    assert request.tool_name == "edit_file"
    assert normalized["snapshot_id"] == read.metadata["snapshot_id"]
    assert normalized["prepared_live_hash"] == read.metadata["raw_bytes_hash"]
    assert "-value = 1" in normalized["diff_preview"]
    assert "+value = 2" in normalized["diff_preview"]
    assert normalized["operations"][0]["new_line_count"] == 1
    assert "new_lines" not in normalized["operations"][0]
    assert edited.metadata["workspace_revision"] == 1
    assert any(event == "edit_approved" for event, _ in events.events)


def test_terminal_edit_approval_shows_readable_operations_and_diff_without_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "main.py"
    path.write_text("value = 1\n", encoding="utf-8")
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=200)
    terminal = TerminalHumanApprover(console, reader=lambda _: "1")
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=terminal)
    registry = create_default_registry(tmp_path, approval_engine=engine)
    read = registry.execute(
        ToolCall(id="read-for-terminal-edit", name="read_file", arguments={"path": "main.py"})
    )
    assert read.ok and read.metadata

    edited = registry.execute(
        ToolCall(
            id="terminal-edit",
            name="edit_file",
            arguments={
                "path": "main.py",
                "snapshot_id": read.metadata["snapshot_id"],
                "operations": [
                    {
                        "op": "replace",
                        "start_line": 1,
                        "end_line": 1,
                        "new_lines": ["value = 2", "print(value)"],
                    }
                ],
            },
        )
    )

    output = buffer.getvalue()
    assert edited.ok
    assert "Edit Approval Required" in output
    assert "Requested edit" in output
    assert "File" in output and "main.py" in output
    assert "Replace lines 1-1 with 2 lines" in output
    assert "Proposed changes" in output
    assert "-value = 1" in output
    assert "+value = 2" in output
    assert "+print(value)" in output
    assert read.metadata["snapshot_id"] not in output
    assert read.metadata["raw_bytes_hash"] not in output
    assert "snapshot_id" not in output
    assert "prepared_live_hash" not in output
    assert "new_lines_sha256" not in output


def test_large_line_deletion_is_classified_as_high_risk(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)

    outcome = engine.authorize(
        tool_name="edit_file",
        arguments={
            "path": "main.py",
            "snapshot_id": "a" * 64,
            "prepared_live_hash": "b" * 64,
            "operations": [{"op": "delete", "start_line": 1, "end_line": 100}],
            "diff_preview": "bounded preview",
        },
        capabilities={Capability.FILESYSTEM_READ, Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    )

    assert outcome.request.assessment.level == RiskLevel.HIGH
    assert any(
        "removes at least 100 lines" in reason for reason in outcome.request.assessment.reasons
    )


def test_always_ask_prompts_even_for_list_files(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALWAYS_ASK, human=human)
    registry = create_default_registry(tmp_path, approval_engine=engine)

    result = registry.execute(ToolCall(id="list-1", name="list_files", arguments={}))

    assert result.ok
    assert [request.tool_name for request in human.requests] == ["list_files"]


def test_always_ask_honors_explicit_matching_repeat_allowance(tmp_path: Path) -> None:
    human = FakeHumanApprover(scope=ApprovalScope.SESSION_EXACT)
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALWAYS_ASK, human=human)
    request = {
        "tool_name": "write_file",
        "arguments": {"path": "file.txt", "content": "value"},
        "capabilities": {Capability.FILESYSTEM_WRITE},
        "session_id": engine.session_id,
    }

    first = engine.authorize(**request)
    repeated = engine.authorize(**request)

    assert first.decision.source == "human"
    assert repeated.decision.source == "session_grant"
    assert len(human.requests) == 1


def test_verification_approval_reviews_the_registered_concrete_command(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    registry = create_default_registry(tmp_path, approval_engine=engine)
    assert registry.verification_runtime is not None
    registry.verification_runtime.bind(memory)
    registry.verification_runtime.register_plan(
        {
            "checks": [
                {
                    "check_id": "smoke",
                    "title": "Run smoke check",
                    "kind": "smoke",
                    "command": {"program": sys.executable, "args": ["-c", "print('ok')"]},
                    "criteria": {"expected_exit_codes": [0]},
                    "required": True,
                    "provenance": "model",
                }
            ]
        }
    )

    result = registry.execute(
        ToolCall(id="verify-1", name="run_verification", arguments={"check_id": "smoke"})
    )

    assert result.ok
    assert len(human.requests) == 1
    request = human.requests[0]
    assert request.tool_name == "run_verification"
    command = request.normalized_arguments["command"]
    assert command["program"] == sys.executable
    assert command["args"] == ["-c", "print('ok')"]


def test_llm_auto_accepts_complete_medium_risk_review(tmp_path: Path) -> None:
    human = FakeHumanApprover(action=ApprovalAction.DENY)
    reviewer = FakeReviewer(
        LLMReviewResult(
            recommendation="allow",
            risk_level="medium",
            task_relevance="required",
            recognized_effects=["process_execution", "execute_project_code"],
            required_constraints=["shell_free_argv", "timeout_60", "workspace_cwd"],
            reason="bounded test command",
            user_prompt=None,
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.action == ApprovalAction.ALLOW
    assert outcome.decision.source == "llm_reviewer"
    assert outcome.decision.constraints == [
        "shell_free_argv",
        "timeout_60",
        "workspace_cwd",
    ]
    assert len(reviewer.requests) == 1
    assert human.requests == []


def test_llm_review_audit_event_records_facts_without_confidence(tmp_path: Path) -> None:
    reviewer = FakeReviewer(
        LLMReviewResult(
            recommendation="allow",
            risk_level="medium",
            task_relevance="helpful",
            recognized_effects=["process_execution", "execute_project_code"],
            required_constraints=["shell_free_argv"],
            reason="The test command is relevant and bounded.",
            user_prompt=None,
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        reviewer=reviewer,
    )
    events = EventCollector()
    engine.event_logger = events

    engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    review_event = next(data for event, data in events.events if event == "llm_approval_reviewed")
    review_started = next(
        data for event, data in events.events if event == "llm_approval_review_started"
    )
    assert review_started["tool"] == "run_command"
    assert review_event["recommendation"] == "allow"
    assert review_event["task_relevance"] == "helpful"
    assert review_event["recognized_effects"] == [
        "process_execution",
        "execute_project_code",
    ]
    assert "confidence" not in review_event
    assert review_event["duration_seconds"] >= 0


def test_invalid_or_unavailable_llm_review_falls_back_to_human(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    reviewer = FakeReviewer(None)
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.source == "human"
    assert len(human.requests) == 1


def test_llm_timeout_falls_back_to_human(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    memory = MemoryState(session_id="approval-test")
    engine = ApprovalEngine(
        mode=ApprovalMode.LLM_AUTO,
        normalizer=RequestNormalizer(tmp_path),
        risk_analyzer=RiskAnalyzer(),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=human,
        llm_reviewer=TimeoutReviewer(),
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.source == "human"
    assert len(human.requests) == 1


def test_openai_reviewer_treats_invalid_json_as_failure() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))]
    )
    completions = SimpleNamespace(create=lambda **_: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    reviewer = OpenAIApprovalReviewer(
        ApiConfig(
            api_key="test-secret",
            base_url="https://example.com/v1",
            model="reviewer",
            context_window_tokens=8_192,
        ),
        client=client,
    )
    request = ApprovalRequest(
        request_id="request",
        session_id="session",
        tool_name="run_command",
        arguments={},
        normalized_arguments={},
        workspace="/workspace",
        fingerprint="fingerprint",
        assessment={"level": RiskLevel.MEDIUM},
    )

    assert reviewer.review(request) is None


def test_openai_reviewer_rejects_legacy_confidence_schema() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "decision": "allow",
                            "risk_level": 2,
                            "confidence": 0.99,
                            "reason": "legacy output",
                            "conditions": [],
                        }
                    )
                )
            )
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    reviewer = OpenAIApprovalReviewer(
        ApiConfig(
            api_key="test-secret",
            base_url="https://example.com/v1",
            model="reviewer",
            context_window_tokens=8_192,
        ),
        client=client,
    )
    request = ApprovalRequest(
        request_id="request",
        session_id="session",
        tool_name="run_command",
        arguments={},
        normalized_arguments={},
        workspace="/workspace",
        fingerprint="fingerprint",
        assessment={"level": RiskLevel.MEDIUM},
    )

    assert reviewer.review(request) is None


def test_llm_cannot_auto_approve_high_risk_network_command(tmp_path: Path) -> None:
    human = FakeHumanApprover(action=ApprovalAction.DENY)
    reviewer = FakeReviewer(
        LLMReviewResult(
            recommendation="allow",
            risk_level="low",
            task_relevance="required",
            recognized_effects=["process_execution", "network_access"],
            reason="unsafe optimistic review",
            user_prompt=None,
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "curl https://example.com", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.assessment.level == RiskLevel.HIGH
    assert outcome.decision.action == ApprovalAction.DENY
    assert len(reviewer.requests) == 1
    assert len(human.requests) == 1
    assert human.llm_reviews == [reviewer.result]


def test_hard_policy_denial_never_calls_llm_reviewer(tmp_path: Path) -> None:
    reviewer = FakeReviewer(None)
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "sudo pytest", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.action == ApprovalAction.DENY
    assert outcome.decision.source == "hard_policy"
    assert reviewer.requests == []


def test_compiler_output_outside_workspace_is_hard_denied_before_llm(tmp_path: Path) -> None:
    reviewer = FakeReviewer(None)
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={
            "command": "g++ main.cpp -o ../app",
            "cwd": ".",
            "timeout_seconds": 60,
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.source == "hard_policy"
    assert outcome.decision.action == ApprovalAction.DENY
    assert Capability.OUTSIDE_WORKSPACE in outcome.request.assessment.capabilities
    assert reviewer.requests == []


def test_llm_deny_is_advice_and_falls_back_to_human(tmp_path: Path) -> None:
    human = FakeHumanApprover(action=ApprovalAction.ALLOW)
    reviewer = FakeReviewer(
        LLMReviewResult(
            recommendation="deny",
            risk_level="high",
            task_relevance="uncertain",
            recognized_effects=["process_execution", "execute_project_code"],
            unknowns=["project test behavior is not known"],
            reason="The command needs a user decision.",
            user_prompt=None,
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.action == ApprovalAction.ALLOW
    assert outcome.decision.source == "human"
    assert human.llm_reviews == [reviewer.result]


def test_llm_allow_with_missing_effect_coverage_falls_back_to_human(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    reviewer = FakeReviewer(
        LLMReviewResult(
            recommendation="allow",
            risk_level="medium",
            task_relevance="required",
            recognized_effects=[],
            reason="The edit appears relevant.",
            user_prompt=None,
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="write_file",
        arguments={"path": "new.py", "content": "value = 1\n"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    )

    assert outcome.decision.source == "human"
    assert outcome.request.facts.explicit_effects == {"filesystem_write"}
    assert human.llm_reviews == [reviewer.result]


def test_llm_allow_with_unknowns_or_unavailable_constraints_falls_back_to_human(
    tmp_path: Path,
) -> None:
    human = FakeHumanApprover()
    reviewer = FakeReviewer(
        LLMReviewResult(
            recommendation="allow",
            risk_level="medium",
            task_relevance="helpful",
            recognized_effects=["process_execution", "execute_project_code"],
            unknowns=["test configuration has not been resolved"],
            required_constraints=["network_isolation"],
            reason="Important facts remain unknown.",
            user_prompt=None,
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.source == "human"
    assert "network_isolation" not in outcome.request.facts.constraints
    assert human.llm_reviews == [reviewer.result]


def test_git_write_cannot_be_auto_approved_by_llm(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    reviewer = FakeReviewer(
        LLMReviewResult(
            recommendation="allow",
            risk_level="medium",
            task_relevance="required",
            recognized_effects=["process_execution", "git_write"],
            reason="The commit is task-related.",
            user_prompt=None,
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "git add app.py", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert "git_write" in outcome.request.facts.explicit_effects
    assert outcome.decision.source == "human"


def test_review_payload_expands_package_script_as_untrusted_semantic_context(
    tmp_path: Path,
) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "npm test", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    payload = build_review_payload(outcome.request)
    stage = payload["execution_plan"]["stages"][0]
    assert stage["semantic_context"] == {
        "analysis_level": "expanded",
        "expanded_command": ["vitest", "run"],
        "reason": ["expanded package test script to vitest"],
    }
    assert "execute_project_code" in outcome.request.facts.explicit_effects


def test_package_installation_facts_force_human_review(tmp_path: Path) -> None:
    human = FakeHumanApprover()
    reviewer = FakeReviewer(
        LLMReviewResult(
            recommendation="allow",
            risk_level="medium",
            task_relevance="required",
            recognized_effects=[
                "process_execution",
                "filesystem_write",
                "network_access",
                "package_installation",
            ],
            reason="Dependencies are required.",
            user_prompt=None,
        )
    )
    engine, _ = create_engine(
        tmp_path,
        mode=ApprovalMode.LLM_AUTO,
        human=human,
        reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "npm install", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert {
        "network_access",
        "package_installation",
        "execute_install_scripts",
    } <= outcome.request.facts.explicit_effects
    assert outcome.decision.source == "human"


def test_openai_reviewer_receives_structured_plan_and_no_output_limit(tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    output = tmp_path / ".reporivet/build/quick_sort"
    output.parent.mkdir(parents=True)
    output.write_text("user-owned", encoding="utf-8")

    def create(**arguments: object) -> object:
        captured.update(arguments)
        content = {
            "recommendation": "ask",
            "risk_level": "medium",
            "task_relevance": "required",
            "recognized_effects": [
                "process_execution",
                "filesystem_read",
                "filesystem_write",
                "compile_workspace_code",
            ],
            "unknowns": ["compiler behavior is not sandboxed"],
            "required_constraints": ["shell_free_argv", "timeout_60"],
            "reason": "Compilation is relevant but needs confirmation.",
            "user_prompt": "Allow compilation of quick_sort.cpp into the build directory?",
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(content)))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    reviewer = OpenAIApprovalReviewer(
        ApiConfig(
            api_key="test-secret",
            base_url="https://example.com/v1",
            model="reviewer",
            context_window_tokens=8_192,
        ),
        client=client,
    )
    human = FakeHumanApprover()
    memory = MemoryState(session_id="approval-test")
    memory.start_task(
        task="implement and verify quick_sort.cpp",
        workspace=str(tmp_path),
        system_prompt="system",
        safety_rules=[],
        completion_rules=[],
        max_steps=10,
    )
    engine = ApprovalEngine(
        mode=ApprovalMode.LLM_AUTO,
        normalizer=RequestNormalizer(tmp_path),
        risk_analyzer=RiskAnalyzer(),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=human,
        llm_reviewer=reviewer,
    )

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={
            "command": "g++ quick_sort.cpp -o .reporivet/build/quick_sort",
            "cwd": ".",
            "timeout_seconds": 60,
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.decision.source == "human"
    assert "max_tokens" not in captured
    messages = captured["messages"]
    assert isinstance(messages, list)
    user_payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert user_payload["task"]["summary"] == "implement and verify quick_sort.cpp"
    assert user_payload["execution_plan"]["stages"][0]["program"] == "g++"
    assert user_payload["deterministic_effects"]["write_paths"] == [
        str((tmp_path / ".reporivet/build/quick_sort").resolve())
    ]
    assert "filesystem_write" in user_payload["deterministic_effects"]["capabilities"]
    assert "shell_free_argv" in user_payload["available_constraints"]
    assert "network_isolation" not in user_payload["available_constraints"]


def test_shell_syntax_is_never_obviously_safe(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest && echo done", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.assessment.level == RiskLevel.HIGH
    assert not outcome.request.assessment.obviously_safe

    registry = create_default_registry(tmp_path, approval_engine=engine)
    rejected = registry.execute(
        ToolCall(
            id="shell-1",
            name="run_command",
            arguments={"command": "pytest && echo done"},
        )
    )
    assert rejected.error_code == "invalid_command"


def test_session_grant_matches_only_exact_normalized_request(tmp_path: Path) -> None:
    human = FakeHumanApprover(scope=ApprovalScope.SESSION_EXACT)
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    common = {
        "tool_name": "write_file",
        "capabilities": {Capability.FILESYSTEM_WRITE},
        "session_id": engine.session_id,
    }

    first = engine.authorize(arguments={"path": "a.py", "content": "one"}, **common)
    repeated = engine.authorize(arguments={"path": "a.py", "content": "one"}, **common)
    changed = engine.authorize(arguments={"path": "a.py", "content": "two"}, **common)

    assert first.decision.source == "human"
    assert repeated.decision.source == "session_grant"
    assert changed.decision.source == "human"
    assert len(human.requests) == 2
    assert len(memory.approval_session_grants) == 2


def test_symlink_change_invalidates_approval_before_execution(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    outcome = engine.authorize(
        tool_name="write_file",
        arguments={"path": "target/file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    )

    target.rmdir()
    target.symlink_to(outside, target_is_directory=True)
    stale = engine.revalidate(outcome)

    assert stale is not None
    assert stale.source == "execution_revalidation"
    assert stale.action == ApprovalAction.DENY


def test_noninteractive_safe_auto_denies_request_needing_human(tmp_path: Path) -> None:
    engine, memory = create_engine(
        tmp_path,
        mode=ApprovalMode.SAFE_AUTO,
        human=NonInteractiveHumanApprover(),
    )

    outcome = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    )

    assert outcome.decision.action == ApprovalAction.DENY
    assert outcome.decision.source == "non_interactive_policy"
    assert outcome.request.fingerprint in memory.denied_request_fingerprints


def test_repeated_once_denial_is_not_prompted_again_during_task(tmp_path: Path) -> None:
    human = FakeHumanApprover(action=ApprovalAction.DENY)
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    request = {
        "tool_name": "write_file",
        "arguments": {"path": "file.txt", "content": "value"},
        "capabilities": {Capability.FILESYSTEM_WRITE},
        "session_id": engine.session_id,
    }

    first = engine.authorize(**request)
    repeated = engine.authorize(**request)

    assert first.decision.source == "human"
    assert repeated.decision.source == "prior_denial"
    assert len(human.requests) == 1


def test_terminal_approval_uses_numbered_options_and_reprompts_invalid_choice(
    tmp_path: Path,
) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    request = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    ).request
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=160)
    choices = iter(["y", "2"])

    decision = TerminalHumanApprover(console, reader=lambda _: next(choices)).ask(request)

    output = buffer.getvalue()
    assert decision.action == ApprovalAction.ALLOW
    assert decision.scope == ApprovalScope.SESSION_EXACT
    assert "1  Allow once" in output
    assert "2  Allow matching repeats for this session" in output
    assert "3  Deny and continue" in output
    assert "4  Stop current run and save session" in output
    assert "Enter a number from 1 to 4" in output
    assert "Requested action" in output
    assert "File" in output and "file.txt" in output
    assert "Content" in output and "5 characters" in output
    assert request.fingerprint[:12] not in output
    assert "sha256" not in output.lower()


def test_terminal_command_approval_uses_readable_fields_without_internal_json(
    tmp_path: Path,
) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    request = engine.authorize(
        tool_name="run_command",
        arguments={"command": "pytest -q", "cwd": "tests", "timeout_seconds": 90},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    ).request
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=180)

    TerminalHumanApprover(console, reader=lambda _: "1").ask(request)

    output = buffer.getvalue()
    assert "Requested action" in output
    assert "Program" in output and "pytest" in output
    assert "Arguments" in output and "-q" in output
    assert "Working directory" in output and "tests" in output
    assert "Timeout" in output and "90 seconds" in output
    assert "Normalized request" not in output
    assert request.fingerprint[:12] not in output


def test_terminal_approval_explains_structured_llm_review(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    request = engine.authorize(
        tool_name="run_command",
        arguments={"command": "npm install", "cwd": ".", "timeout_seconds": 60},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    ).request
    review = LLMReviewResult(
        recommendation="ask",
        risk_level="high",
        task_relevance="helpful",
        recognized_effects=["process_execution", "network_access", "package_installation"],
        unknowns=["third-party install scripts are unknown"],
        required_constraints=["shell_free_argv", "timeout_60"],
        reason="Dependency installation has external effects.",
        user_prompt="Allow network access and third-party install scripts?",
    )
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=200)

    TerminalHumanApprover(console, reader=lambda _: "1").ask(request, llm_review=review)

    output = buffer.getvalue()
    assert "ASK · risk high · relevance helpful" in output
    assert "Effects: process_execution, network_access, package_installation" in output
    assert "Unknowns: third-party install scripts are unknown" in output
    assert "Constraints: shell_free_argv, timeout_60" in output
    assert "Approval question: Allow network access and third-party install scripts?" in output
    assert "confidence" not in output.lower()


def test_terminal_option_four_stops_current_run(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    request = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    ).request
    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    decision = TerminalHumanApprover(console, reader=lambda _: "4").ask(request)

    assert decision.action == ApprovalAction.DENY
    assert decision.abort_agent
    assert "session will be saved" in decision.reason


def test_terminal_denial_can_include_direction_for_agent(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    request = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    ).request
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=180)
    answers = iter(["3", "先读取现有文件，只修改目标函数"])

    decision = TerminalHumanApprover(console, reader=lambda _: next(answers)).ask(request)

    assert decision.action == ApprovalAction.DENY
    assert decision.scope == ApprovalScope.ONCE
    assert decision.guidance == "先读取现有文件，只修改目标函数"
    assert "Deny and continue" in buffer.getvalue()


def test_terminal_denial_reason_is_optional(tmp_path: Path) -> None:
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.ALLOW_ALL)
    request = engine.authorize(
        tool_name="write_file",
        arguments={"path": "file.txt", "content": "value"},
        capabilities={Capability.FILESYSTEM_WRITE},
        session_id=engine.session_id,
    ).request
    answers = iter(["3", ""])

    decision = TerminalHumanApprover(
        Console(file=StringIO(), force_terminal=False, color_system=None),
        reader=lambda _: next(answers),
    ).ask(request)

    assert decision.action == ApprovalAction.DENY
    assert not decision.abort_agent
    assert decision.guidance is None


def test_guided_denial_is_returned_to_model_and_reused_for_exact_retry(tmp_path: Path) -> None:
    guidance = "Use edit_file on app.py instead of overwriting the file"
    human = FakeHumanApprover(action=ApprovalAction.DENY, guidance=guidance)
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    events = EventCollector()
    engine.event_logger = events
    registry = create_default_registry(tmp_path, approval_engine=engine)
    call = ToolCall(
        id="write-1",
        name="write_file",
        arguments={"path": "app.py", "content": "replacement"},
    )

    first = registry.execute(call)
    repeated = registry.execute(call)

    assert not first.ok and not repeated.ok
    assert f"User direction: {guidance}" in (first.error or "")
    assert f"User direction: {guidance}" in (repeated.error or "")
    assert first.metadata and first.metadata["approval_guidance"] == guidance
    assert len(human.requests) == 1
    assert list(memory.approval_denial_guidance.values()) == [guidance]
    decision_events = [data for event, data in events.events if event == "approval_decided"]
    assert decision_events[0]["guidance"] == guidance


def test_approval_events_cover_request_decision_and_execution(tmp_path: Path) -> None:
    memory = MemoryState(session_id="approval-test")
    events = EventCollector()
    engine = ApprovalEngine(
        mode=ApprovalMode.ALLOW_ALL,
        normalizer=RequestNormalizer(tmp_path),
        risk_analyzer=RiskAnalyzer(),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=FakeHumanApprover(),
        event_logger=events,
    )
    registry = create_default_registry(tmp_path, approval_engine=engine)

    result = registry.execute(ToolCall(id="list-1", name="list_files", arguments={}))

    assert result.ok
    assert [event[0] for event in events.events] == [
        "approval_requested",
        "approval_decided",
        "approved_tool_started",
        "approved_tool_executed",
    ]


def test_exact_bounded_build_is_auto_approved_at_medium_risk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    (tmp_path / "snake.cpp").write_text("int main() {}\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "g++ -o snake snake.cpp", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.assessment.level == RiskLevel.MEDIUM
    assert outcome.request.facts.operation_class == OperationClass.BUILD
    assert outcome.request.facts.analysis_level == AnalysisLevel.EXACT
    assert outcome.decision.source == "semantic_template:bounded_build"
    assert human.requests == []


@pytest.mark.parametrize(
    ("command", "operation"),
    [
        ("reporivet skill list", OperationClass.READ),
        ("reporivet skill show sample-skill", OperationClass.READ),
        ("reporivet skill validate draft/SKILL.md", OperationClass.STATIC_CHECK),
    ],
)
def test_trusted_reporivet_skill_inspection_is_auto_approved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    operation: OperationClass,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    draft = tmp_path / "draft" / "SKILL.md"
    draft.parent.mkdir()
    draft.write_text("# draft\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": command, "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.facts.operation_class == operation
    assert outcome.request.facts.analysis_level == AnalysisLevel.EXACT
    assert outcome.request.facts.executable_origin == ExecutableOrigin.TRUSTED_TOOLCHAIN
    assert outcome.decision.source == "semantic_template:reporivet_skill_inspection"
    assert human.requests == []


def test_trusted_reporivet_skill_generation_is_auto_approved_only_for_new_workspace_drafts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    source = tmp_path / "foreign" / "SKILL.md"
    source.parent.mkdir()
    source.write_text("# source\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    initialized = engine.authorize(
        tool_name="run_command",
        arguments={
            "command": "reporivet skill init generated-skill --output drafts",
            "cwd": ".",
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    converted = engine.authorize(
        tool_name="run_command",
        arguments={
            "command": (
                "reporivet skill convert foreign/SKILL.md --id converted-skill --output drafts"
            ),
            "cwd": ".",
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert initialized.request.facts.operation_class == OperationClass.GENERATE
    assert converted.request.facts.operation_class == OperationClass.GENERATE
    assert initialized.request.facts.write_paths == [
        str((tmp_path / "drafts/generated-skill/SKILL.md").resolve())
    ]
    assert converted.request.facts.read_paths == [str(source.resolve())]
    assert converted.request.facts.write_paths == [
        str((tmp_path / "drafts/converted-skill/SKILL.md").resolve())
    ]
    assert initialized.decision.source == "semantic_template:reporivet_skill_generation"
    assert converted.decision.source == "semantic_template:reporivet_skill_generation"
    assert human.requests == []


def test_reporivet_skill_global_changes_and_untrusted_executables_are_not_auto_approved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    draft = tmp_path / "draft" / "SKILL.md"
    draft.parent.mkdir()
    draft.write_text("# draft\n", encoding="utf-8")
    shadow = tmp_path / "reporivet"
    shadow.write_text("#!/bin/sh\n", encoding="utf-8")
    shadow.chmod(0o755)
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    install = engine.authorize(
        tool_name="run_command",
        arguments={"command": "reporivet skill install draft/SKILL.md", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    untrusted = engine.authorize(
        tool_name="run_command",
        arguments={"command": "./reporivet skill validate draft/SKILL.md", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert install.request.facts.analysis_level == AnalysisLevel.OPAQUE
    assert install.decision.source == "human"
    assert untrusted.request.facts.executable_origin == ExecutableOrigin.WORKSPACE
    assert untrusted.request.facts.analysis_level == AnalysisLevel.OPAQUE
    assert untrusted.decision.source == "human"
    assert len(human.requests) == 2


def test_reporivet_skill_generation_does_not_auto_approve_overwrite_or_workspace_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    existing = tmp_path / "drafts" / "existing-skill" / "SKILL.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("# user draft\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    overwrite = engine.authorize(
        tool_name="run_command",
        arguments={
            "command": "reporivet skill init existing-skill --output drafts",
            "cwd": ".",
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    escape = engine.authorize(
        tool_name="run_command",
        arguments={
            "command": "reporivet skill init escaped-skill --output ../outside",
            "cwd": ".",
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert overwrite.request.facts.overwrites_existing
    assert overwrite.decision.source == "human"
    assert escape.request.facts.outside_workspace
    assert escape.decision.source == "hard_policy"
    assert len(human.requests) == 1


def test_workspace_compiler_with_trusted_name_is_not_auto_approved(tmp_path: Path) -> None:
    compiler = tmp_path / "g++"
    compiler.write_text("#!/bin/sh\n", encoding="utf-8")
    compiler.chmod(0o755)
    (tmp_path / "snake.cpp").write_text("int main() {}\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "./g++ -o snake snake.cpp", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.facts.executable_origin == ExecutableOrigin.WORKSPACE
    assert outcome.decision.source == "human"


def test_build_that_overwrites_unknown_file_requires_human(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    (tmp_path / "snake.cpp").write_text("int main() {}\n", encoding="utf-8")
    (tmp_path / "snake").write_text("user data", encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    outcome = engine.authorize(
        tool_name="run_command",
        arguments={"command": "g++ -o snake snake.cpp", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.facts.overwrites_existing
    assert set(item.value for item in outcome.request.facts.output_provenance.values()) == {
        "user_file"
    }
    assert outcome.decision.source == "human"


def test_build_response_file_and_plugin_flags_remain_opaque(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    response_file = engine.authorize(
        tool_name="run_command",
        arguments={"command": "g++ @build.rsp", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    plugin = engine.authorize(
        tool_name="run_command",
        arguments={"command": "g++ -fplugin=custom.so -o snake snake.cpp", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert response_file.request.facts.analysis_level == AnalysisLevel.OPAQUE
    assert plugin.request.facts.analysis_level == AnalysisLevel.OPAQUE
    assert response_file.decision.source == plugin.decision.source == "human"


def test_registered_pytest_check_matches_bounded_test_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    human = FakeHumanApprover()
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    memory.verification_plan = VerificationPlan(
        plan_id="verify-tests",
        checks=[
            VerificationCheck(
                check_id="tests",
                title="Tests",
                kind=VerificationKind.TEST,
                command=CommandSpec(program="pytest", args=["-q"]),
                provenance="model",
            )
        ],
    )

    outcome = engine.authorize(
        tool_name="run_verification",
        arguments={
            "check_id": "tests",
            "command": "pytest -q",
            "cwd": ".",
            "timeout_seconds": 60,
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.facts.verification_kind == "test"
    assert outcome.decision.source == "semantic_template:bounded_test"
    assert human.requests == []


def test_registered_npm_test_is_expanded_before_template_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run"}}),
        encoding="utf-8",
    )
    human = FakeHumanApprover()
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    memory.verification_plan = VerificationPlan(
        plan_id="verify-js",
        checks=[
            VerificationCheck(
                check_id="js-tests",
                title="JavaScript tests",
                kind=VerificationKind.TEST,
                command=CommandSpec(program="npm", args=["test"]),
                provenance="model",
            )
        ],
    )

    outcome = engine.authorize(
        tool_name="run_verification",
        arguments={
            "check_id": "js-tests",
            "command": "npm test",
            "cwd": ".",
            "timeout_seconds": 60,
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.facts.analysis_level == AnalysisLevel.EXPANDED
    assert outcome.request.facts.expanded_command == ["vitest", "run"]
    assert outcome.decision.source == "semantic_template:bounded_test"


def test_chained_package_test_script_remains_opaque(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run && curl https://example.test"}}),
        encoding="utf-8",
    )
    human = FakeHumanApprover()
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    memory.verification_plan = VerificationPlan(
        plan_id="verify-js",
        checks=[
            VerificationCheck(
                check_id="js-tests",
                title="JavaScript tests",
                kind=VerificationKind.TEST,
                command=CommandSpec(program="npm", args=["test"]),
                provenance="model",
            )
        ],
    )

    outcome = engine.authorize(
        tool_name="run_verification",
        arguments={
            "check_id": "js-tests",
            "command": "npm test",
            "cwd": ".",
            "timeout_seconds": 60,
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert outcome.request.facts.analysis_level == AnalysisLevel.OPAQUE
    assert outcome.decision.source == "human"


def test_generation_is_auto_approved_only_for_managed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    (tmp_path / "schema.proto").write_text('syntax = "proto3";\n', encoding="utf-8")
    human = FakeHumanApprover()
    engine, _ = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)

    managed = engine.authorize(
        tool_name="run_command",
        arguments={
            "command": "protoc --python_out=.reporivet/generated schema.proto",
            "cwd": ".",
        },
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    source_directory = engine.authorize(
        tool_name="run_command",
        arguments={"command": "protoc --python_out=. schema.proto", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert managed.decision.source == "semantic_template:managed_generation"
    assert source_directory.decision.source == "human"


def test_session_artifact_is_auto_approved_until_workspace_revision_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "repo_rivet.approval.semantic_analyzer.shutil.which", lambda _: sys.executable
    )
    source = tmp_path / "snake.cpp"
    source.write_text("int main() {}\n", encoding="utf-8")
    human = FakeHumanApprover()
    engine, memory = create_engine(tmp_path, mode=ApprovalMode.SAFE_AUTO, human=human)
    build = engine.authorize(
        tool_name="run_command",
        arguments={"command": "g++ -o snake snake.cpp", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    artifact = tmp_path / "snake"
    artifact.write_bytes(b"current artifact")
    artifact.chmod(0o755)
    engine.record_execution(build, ok=True, metadata={})

    first_run = engine.authorize(
        tool_name="run_command",
        arguments={"command": "./snake", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    unsafe_arguments = engine.authorize(
        tool_name="run_command",
        arguments={"command": "./snake ../user-data", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    memory.workspace_revision += 1
    stale_run = engine.authorize(
        tool_name="run_command",
        arguments={"command": "./snake", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    stale_rebuild = engine.authorize(
        tool_name="run_verification",
        arguments={"check_id": "build", "command": "g++ -o snake snake.cpp", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )
    artifact.write_bytes(b"manually replaced")
    tampered_rebuild = engine.authorize(
        tool_name="run_verification",
        arguments={"check_id": "build", "command": "g++ -o snake snake.cpp", "cwd": "."},
        capabilities={Capability.PROCESS_EXECUTE},
        session_id=engine.session_id,
    )

    assert "snake" in memory.artifact_registry
    assert first_run.decision.source == "semantic_template:session_artifact_run"
    assert unsafe_arguments.decision.source == "human"
    assert stale_run.request.facts.executable_origin == ExecutableOrigin.WORKSPACE
    assert stale_run.decision.source == "human"
    assert set(stale_rebuild.request.facts.output_provenance.values()) == {ArtifactProvenance.STALE}
    assert stale_rebuild.decision.source == "semantic_template:bounded_build"
    assert set(tampered_rebuild.request.facts.output_provenance.values()) == {
        ArtifactProvenance.USER_FILE
    }
    assert tampered_rebuild.decision.source == "human"


def test_registered_build_can_overwrite_stale_current_session_artifact(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "snake.cpp").write_text("int main() {}\n", encoding="utf-8")
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    compiler = toolchain / "g++"
    compiler.write_text('#!/bin/sh\nprintf "artifact" > "$2"\n', encoding="utf-8")
    compiler.chmod(0o755)

    memory = MemoryState(session_id="approval-test")
    memory.verification_plan = VerificationPlan(
        plan_id="verify-build",
        checks=[
            VerificationCheck(
                check_id="build",
                title="Build snake",
                kind=VerificationKind.BUILD,
                command=CommandSpec(
                    program=str(compiler),
                    args=["-o", "snake", "snake.cpp"],
                ),
                provenance="model",
            )
        ],
    )
    human = FakeHumanApprover()
    events = EventCollector()
    engine = ApprovalEngine(
        mode=ApprovalMode.SAFE_AUTO,
        normalizer=RequestNormalizer(workspace),
        risk_analyzer=RiskAnalyzer(
            ApprovalFactAnalyzer(trusted_executable_directories=[str(toolchain)])
        ),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=human,
        event_logger=events,
    )
    registry = create_default_registry(workspace, approval_engine=engine)
    assert registry.verification_runtime is not None
    registry.verification_runtime.bind(memory)
    call = ToolCall(id="verify-build", name="run_verification", arguments={"check_id": "build"})

    first = registry.execute(call)
    memory.workspace_revision += 1
    second = registry.execute(call)

    assert first.ok and second.ok
    assert "snake" in memory.artifact_registry
    assert human.requests == []
    sources = [data["source"] for event, data in events.events if event == "approval_decided"]
    assert sources == [
        "semantic_template:bounded_build",
        "semantic_template:bounded_build",
    ]
