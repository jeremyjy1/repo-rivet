import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repo_rivet.llm.base import ModelRequestOptions, ModelResponse
from repo_rivet.memory.models import MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.reasoning.models import ObservationEvent
from repo_rivet.subagents.manager import SubagentManager
from repo_rivet.subagents.models import DelegateTaskArguments, SubagentProfile
from repo_rivet.subagents.policy import ScopedWorkspacePathPolicy, profile_runtime_config
from repo_rivet.subagents.tools import ReadToolOutputTool, ReadVerificationResultTool
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.verification.models import VerificationResult, VerificationStatus


class RecordingEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log(self, event_type: str, **data: Any) -> None:
        self.events.append((event_type, data))


class EvidenceReportingModel:
    def __init__(self, *, invalid_evidence: bool = False) -> None:
        self.calls = 0
        self.invalid_evidence = invalid_evidence

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: ModelRequestOptions | None = None,
    ) -> ModelResponse:
        del options
        names = {str(tool["function"]["name"]) for tool in tools}
        assert "edit_file" not in names
        assert "write_file" not in names
        assert "run_command" not in names
        assert "delegate_task" not in names
        self.calls += 1
        if self.calls % 2 == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id=f"read-{self.calls}",
                        name="read_file",
                        arguments={"path": "src/app.py", "start_line": 1, "end_line": 20},
                    )
                ],
                finish_reason="tool_calls",
            )
        payload = json.loads(str(messages[-1]["content"]))
        evidence_ref = (
            "obs-does-not-exist"
            if self.invalid_evidence
            else str(payload["metadata"]["evidence_ref"])
        )
        task = next(
            json.loads(str(message["content"]))
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        report = {
            "delegation_id": task["delegation_id"],
            "status": "completed",
            "summary": "The implementation entry point is src/app.py.",
            "findings": [
                {
                    "statement": "src/app.py contains the inspected entry point.",
                    "evidence_refs": [evidence_ref],
                    "affected_paths": ["src/app.py"],
                    "importance": "high",
                }
            ],
            "recommended_actions": ["Use the observed entry point in the parent decision."],
            "base_workspace_revision": task["base_workspace_revision"],
        }
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    id=f"report-{self.calls}",
                    name="submit_subagent_report",
                    arguments={"report": report},
                )
            ],
            finish_reason="tool_calls",
        )


class BlockingReportModel:
    def __init__(self, started: threading.Barrier, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: ModelRequestOptions | None = None,
    ) -> ModelResponse:
        del tools, options
        task = next(
            json.loads(str(message["content"]))
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        self.started.wait(timeout=2)
        self.release.wait(timeout=2)
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    id="report",
                    name="submit_subagent_report",
                    arguments={
                        "report": {
                            "delegation_id": task["delegation_id"],
                            "status": "completed",
                            "summary": "Scoped inspection completed without file reads.",
                            "findings": [],
                            "base_workspace_revision": task["base_workspace_revision"],
                        }
                    },
                )
            ],
            finish_reason="tool_calls",
        )


def _manager(
    tmp_path: Path,
    model_factory: Any,
    *,
    max_concurrency: int = 2,
) -> tuple[SubagentManager, MemoryState, RecordingEvents]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "src").mkdir(exist_ok=True)
    (workspace / "src" / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    events = RecordingEvents()
    manager = SubagentManager(
        workspace=workspace,
        parent_store=MemoryStore(tmp_path / "parent-session"),
        model_client_factory=model_factory,
        event_logger=events,
        max_concurrency=max_concurrency,
    )
    memory = MemoryState(session_id="parent", workspace_revision=3)
    manager.bind(memory)
    return manager, memory, events


def _delegation(objective: str = "Locate the entry point") -> DelegateTaskArguments:
    return DelegateTaskArguments(
        profile=SubagentProfile.EXPLORER,
        objective=objective,
        deliverable="Return the entry file and evidence",
        scope_paths=["src"],
    )


def test_read_only_profiles_never_expose_mutation_or_nested_delegation() -> None:
    expected = {
        SubagentProfile.EXPLORER: {
            "list_files",
            "read_file",
            "search_text",
            "semantic_query",
            "git_diff",
            "submit_subagent_report",
        },
        SubagentProfile.TEST_ANALYST: {
            "read_file",
            "search_text",
            "semantic_query",
            "read_tool_output",
            "submit_subagent_report",
        },
        SubagentProfile.REVIEWER: {
            "read_file",
            "semantic_query",
            "git_diff",
            "read_verification_result",
            "submit_subagent_report",
        },
    }
    for profile in SubagentProfile:
        tools = profile_runtime_config(profile, ["src"]).allowed_tools
        assert tools == expected[profile]
        assert "submit_subagent_report" in tools
        assert "edit_file" not in tools
        assert "write_file" not in tools
        assert "delete_path" not in tools
        assert "run_command" not in tools
        assert "run_verification" not in tools
        assert "delegate_task" not in tools
        assert "finish_task" not in tools


def test_parent_evidence_readers_only_expose_explicit_delegated_facts(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "parent-session")
    memory = MemoryState(session_id="parent")
    output_ref = store.save_tool_output(
        ToolCall(id="command-1", name="run_command", arguments={}),
        ToolResult(ok=True, output="short", raw_output="complete test output"),
        step=1,
    )
    assert output_ref is not None
    memory.observation_events.append(
        ObservationEvent(
            event_id="obs-command",
            session_id="parent",
            step=1,
            tool_call_id="command-1",
            tool_name="run_command",
            ok=True,
            result_summary="Command passed.",
            output_ref=output_ref,
        )
    )
    now = datetime.now(UTC)
    memory.verification_results["tests"] = VerificationResult(
        check_id="tests",
        status=VerificationStatus.PASSED,
        workspace_revision=1,
        exit_code=0,
        reasons=["all checks passed"],
        stdout_ref="command_outputs/private.log",
        started_at=now,
        finished_at=now,
    )

    output_tool = ReadToolOutputTool(store, memory, ["obs-command"])
    output_result = output_tool.execute({"evidence_ref": "obs-command"})
    verification_tool = ReadVerificationResultTool(memory, ["tests"])
    verification_result = verification_tool.execute({"check_id": "tests"})

    assert output_result.ok
    assert output_result.output == "complete test output"
    assert verification_result.ok
    assert json.loads(verification_result.output)["status"] == "passed"
    assert "stdout_ref" not in verification_result.output
    denied = output_tool.execute({"evidence_ref": "obs-not-delegated"})
    assert not denied.ok
    assert "not delegated" in str(denied.error)


def test_scoped_path_policy_rejects_files_outside_delegation(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    policy = ScopedWorkspacePathPolicy(tmp_path, allowed_paths=["src"])

    assert policy.relative("src").as_posix() == "src"
    try:
        policy.resolve("tests/test_app.py")
    except ValueError as error:
        assert "outside the delegated scope" in str(error)
    else:  # pragma: no cover - explicit safety assertion
        raise AssertionError("Out-of-scope path was accepted")


def test_subagent_runs_in_isolated_store_and_reuses_fresh_report(tmp_path: Path) -> None:
    model = EvidenceReportingModel()
    manager, memory, events = _manager(tmp_path, lambda _events: model)

    first = manager.delegate(_delegation())
    second = manager.delegate(_delegation())

    assert first.ok and second.ok
    assert model.calls == 2
    assert json.loads(second.output)["reused"] is True
    assert memory.modified_files == set()
    child_directories = list((tmp_path / "parent-session" / "subagents").iterdir())
    assert len(child_directories) == 1
    assert (child_directories[0] / "state.json").is_file()
    assert (child_directories[0] / "events.jsonl").is_file()
    assert (child_directories[0] / "report.json").is_file()
    record = json.loads((child_directories[0] / "record.json").read_text(encoding="utf-8"))
    assert record["child_run_id"]
    assert any(event == "subagent_report_accepted" for event, _data in events.events)


def test_changed_snapshot_prevents_report_reuse(tmp_path: Path) -> None:
    model = EvidenceReportingModel()
    manager, _memory, _events = _manager(tmp_path, lambda _events: model)

    assert manager.delegate(_delegation()).ok
    (tmp_path / "workspace" / "src" / "app.py").write_text(
        "def main():\n    return 2\n",
        encoding="utf-8",
    )
    refreshed = manager.delegate(_delegation())

    assert refreshed.ok
    assert model.calls == 4
    assert json.loads(refreshed.output)["reused"] is False


def test_unknown_report_evidence_is_rejected(tmp_path: Path) -> None:
    model = EvidenceReportingModel(invalid_evidence=True)
    manager, _memory, _events = _manager(tmp_path, lambda _events: model)

    result = manager.delegate(_delegation())

    assert not result.ok
    assert result.error_code == "subagent_report_invalid"
    assert "unknown evidence" in str(result.error)


def test_subagent_manager_enforces_two_child_concurrency_limit(tmp_path: Path) -> None:
    started = threading.Barrier(3)
    release = threading.Event()
    manager, _memory, _events = _manager(
        tmp_path,
        lambda _events: BlockingReportModel(started, release),
    )
    results: list[Any] = []
    first = threading.Thread(
        target=lambda: results.append(manager.delegate(_delegation("Inspect first module")))
    )
    second = threading.Thread(
        target=lambda: results.append(manager.delegate(_delegation("Inspect second module")))
    )
    first.start()
    second.start()
    started.wait(timeout=2)

    third = manager.delegate(_delegation("Inspect third module"))
    release.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not third.ok
    assert third.error_code == "subagent_concurrency_limit"
    assert len(results) == 2
    assert all(result.ok for result in results)


def test_matching_inflight_delegation_waits_and_reuses_one_child(tmp_path: Path) -> None:
    started = threading.Barrier(2)
    release = threading.Event()
    factory_calls = 0
    factory_lock = threading.Lock()

    def factory(_events: Any) -> BlockingReportModel:
        nonlocal factory_calls
        with factory_lock:
            factory_calls += 1
        return BlockingReportModel(started, release)

    manager, _memory, _events = _manager(tmp_path, factory)
    results: list[Any] = []
    first = threading.Thread(target=lambda: results.append(manager.delegate(_delegation())))
    second = threading.Thread(target=lambda: results.append(manager.delegate(_delegation())))
    first.start()
    started.wait(timeout=2)
    second.start()
    release.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert factory_calls == 1
    assert len(results) == 2
    assert all(result.ok for result in results)
    assert sorted(json.loads(result.output)["reused"] for result in results) == [False, True]
