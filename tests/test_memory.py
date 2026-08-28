import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_rivet.editing.runtime import EditingRuntime
from repo_rivet.editing.tools import EditFileTool
from repo_rivet.memory.context_manager import SYSTEM_PROMPT
from repo_rivet.memory.models import MemoryConfig, MemoryState, Message
from repo_rivet.memory.store import MemoryStore
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.tools.filesystem import ReadFileTool
from repo_rivet.verification.models import (
    CommandSpec,
    VerificationCheck,
    VerificationKind,
    VerificationPlan,
    VerificationResult,
    VerificationStatus,
)


def make_memory(session_id: str = "test-session") -> MemoryState:
    memory = MemoryState(session_id=session_id)
    memory.start_task(
        task="original task must remain",
        workspace="/workspace",
        system_prompt=SYSTEM_PROMPT,
        safety_rules=["stay in workspace"],
        completion_rules=["verify changes"],
        max_steps=30,
    )
    return memory


def read_call() -> ToolCall:
    return ToolCall(id="read-1", name="read_file", arguments={"path": "src/app.py"})


def read_result(content: str, sha256: str = "abc123") -> ToolResult:
    return ToolResult(
        ok=True,
        output=f"1 | {content}",
        metadata={"path": "src/app.py", "sha256": sha256},
        raw_output=content,
    )


def verification_plan() -> VerificationPlan:
    return VerificationPlan(
        plan_id="plan-1",
        checks=[
            VerificationCheck(
                check_id="tests",
                title="Run tests",
                kind=VerificationKind.TEST,
                command=CommandSpec(program="pytest", args=["-q"]),
                provenance="model",
            )
        ],
    )


def verification_result(
    status: VerificationStatus,
    *,
    revision: int,
) -> VerificationResult:
    now = datetime.now(UTC)
    return VerificationResult(
        check_id="tests",
        status=status,
        workspace_revision=revision,
        exit_code=0 if status == VerificationStatus.PASSED else 1,
        reasons=[f"check {status.value}"],
        started_at=now,
        finished_at=now,
    )


def test_repeated_unchanged_file_read_still_returns_requested_content() -> None:
    memory = make_memory()

    memory.record_tool_result(read_call(), read_result("large file content"), step=1)
    memory.record_tool_result(read_call(), read_result("large file content"), step=2)

    latest_message = json.loads(memory.messages[-1].content or "{}")
    assert latest_message["output"] == "1 | large file content"
    assert memory.file_memories["src/app.py"].last_read_step == 2


def test_file_modification_invalidates_previous_file_memory() -> None:
    memory = make_memory()
    memory.record_tool_result(read_call(), read_result("old content"), step=1)
    write = ToolCall(
        id="write-1",
        name="edit_file",
        arguments={
            "path": "src/app.py",
            "snapshot_id": "a" * 64,
            "operations": [
                {
                    "op": "replace",
                    "start_line": 1,
                    "end_line": 1,
                    "new_lines": ["new"],
                }
            ],
        },
    )

    memory.record_tool_result(
        write,
        ToolResult(ok=True, output="replaced", metadata={"sha256": "new-hash"}),
        step=2,
    )

    assert "src/app.py" not in memory.file_memories
    assert "src/app.py" in memory.invalidated_files
    assert memory.workspace_revision == 1
    assert "src/app.py" in memory.summary.files_modified


def test_task_update_never_overwrites_original_task() -> None:
    memory = make_memory()

    memory.start_task(
        task="also add unit tests",
        workspace="/workspace",
        system_prompt=SYSTEM_PROMPT,
        safety_rules=[],
        completion_rules=[],
        max_steps=30,
    )

    assert memory.fixed and memory.fixed.original_task == "original task must remain"
    assert memory.task_updates == ["also add unit tests"]
    specification = memory.task_specification()
    assert "original task must remain" in specification
    assert "also add unit tests" in specification


def test_failed_verification_is_preserved_in_structured_summary() -> None:
    memory = make_memory()
    call = ToolCall(id="command-1", name="run_verification", arguments={"check_id": "tests"})
    explicit_result = verification_result(VerificationStatus.FAILED, revision=0)
    result = ToolResult(
        ok=True,
        output="Exit code: 1\nFAILED tests/test_app.py",
        metadata={
            "exit_code": 1,
            "timed_out": False,
            "command": "pytest -q",
            "verification_result": explicit_result.model_dump(mode="json"),
        },
        raw_output="Exit code: 1\nFAILED tests/test_app.py\n1 failed",
    )

    memory.record_tool_result(
        call,
        result,
        step=3,
        full_output_path="command_outputs/step-3.log",
    )

    assert memory.summary.verification_status == "failed: tests"
    assert any("Verification tests failed" in issue for issue in memory.summary.unresolved_issues)
    assert memory.command_outputs[-1].full_output_path == "command_outputs/step-3.log"
    assert "retained in session audit storage" in memory.command_outputs[-1].context_output
    assert "command_outputs/step-3.log" not in memory.command_outputs[-1].context_output
    assert memory.command_outputs[-1].original_chars == len(result.raw_output or "")
    assert memory.command_outputs[-1].estimated_tokens > 0


def test_model_message_hides_inaccessible_session_output_reference() -> None:
    message = Message(
        role="tool",
        tool_call_id="read-1",
        content=json.dumps(
            {
                "ok": True,
                "output": "content\nFull output: file_snapshots/step-1.txt",
                "metadata": {
                    "evidence_ref": "obs-1",
                    "output_ref": "file_snapshots/step-1.txt",
                },
            }
        ),
    )

    visible = json.loads(message.as_chat_message()["content"])

    assert visible["output"] == "content"
    assert visible["metadata"] == {"evidence_ref": "obs-1"}


def test_process_observation_is_stored_separately_from_verification() -> None:
    memory = make_memory()
    call = ToolCall(id="command-1", name="run_command", arguments={"command": "tool --check"})

    memory.record_tool_result(
        call,
        ToolResult(
            ok=True,
            output="Exit code: 0",
            metadata={
                "exit_code": 0,
                "command": "tool --check",
                "process_observation": {
                    "command_id": "process-1",
                    "argv": ["tool", "--check"],
                    "cwd": "/workspace",
                    "exit_code": 0,
                    "timed_out": False,
                    "duration_ms": 10,
                },
            },
        ),
        step=1,
        full_output_path="command_outputs/step-1.log",
    )

    observation = memory.process_observations[-1]
    assert observation.exit_code == 0
    assert observation.stdout_ref == "command_outputs/step-1.log"
    assert memory.verification_results == {}


def test_new_modification_invalidates_passed_verification() -> None:
    memory = make_memory()
    memory.verification_plan = verification_plan()
    memory.verification_results["tests"] = verification_result(
        VerificationStatus.PASSED,
        revision=0,
    )

    write = ToolCall(id="write-1", name="write_file", arguments={"path": "new.py"})
    memory.record_tool_result(write, ToolResult(ok=True, output="written"), step=2)

    assert memory.workspace_revision == 1
    assert memory.verification_results["tests"].status == VerificationStatus.STALE


def test_memory_store_round_trip_and_full_output_persistence(tmp_path: Path) -> None:
    store = MemoryStore.create(tmp_path, secrets=("secret-value",))
    memory = make_memory(store.session_id)
    call = ToolCall(id="command/1", name="run_command", arguments={"command": "pytest -q"})
    result = ToolResult(
        ok=True,
        output="short",
        raw_output="complete\nsecret-value\noutput",
    )

    output_ref = store.save_tool_output(call, result, step=4)
    store.log("tool_result", api_key="must-not-leak", output_ref=output_ref)
    store.save_state(memory, status="running", agent_step=4)
    restored = store.load_state()

    assert output_ref is not None
    persisted_output = (store.session_dir / output_ref).read_text(encoding="utf-8")
    assert "secret-value" not in persisted_output
    assert "[REDACTED]" in persisted_output
    assert restored.fixed and restored.fixed.original_task == "original task must remain"
    assert restored.session_id == store.session_id
    assert "must-not-leak" not in store.events_path.read_text(encoding="utf-8")


def test_provider_reasoning_and_ephemeral_continuation_are_never_persisted(
    tmp_path: Path,
) -> None:
    store = MemoryStore.create(tmp_path)
    memory = make_memory(store.session_id)
    memory.messages.append(
        Message(
            role="assistant",
            content=None,
            reasoning_content="hidden provider state",
            ephemeral=True,
        )
    )
    memory.append_ephemeral_system("continue truncated output", step=0)

    store.save_state(memory, status="running")

    state_text = store.state_path.read_text(encoding="utf-8")
    restored = store.load_state()
    assert "hidden provider state" not in state_text
    assert "continue truncated output" not in state_text
    assert len(restored.messages) == len(memory.messages) - 2


def test_resume_detects_external_file_change(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "src"
    source.mkdir()
    file_path = source / "app.py"
    file_path.write_text("old", encoding="utf-8")
    store = MemoryStore.create(tmp_path / "sessions")
    memory = MemoryState(session_id=store.session_id)
    memory.start_task(
        task="task",
        workspace=str(workspace.resolve()),
        system_prompt=SYSTEM_PROMPT,
        safety_rules=[],
        completion_rules=[],
        max_steps=30,
    )
    old_hash = hashlib.sha256(b"old").hexdigest()
    memory.record_tool_result(read_call(), read_result("old", old_hash), step=1)
    memory.verification_plan = verification_plan()
    memory.verification_results["tests"] = verification_result(
        VerificationStatus.PASSED,
        revision=0,
    )
    memory.summary.verification_status = "passed: pytest -q"
    file_path.write_text("changed externally", encoding="utf-8")

    changed = store.validate_workspace(memory, workspace)

    assert changed == ["src/app.py"]
    assert "src/app.py" in memory.invalidated_files
    assert memory.workspace_revision == 1
    assert memory.verification_results["tests"].status == VerificationStatus.STALE
    assert memory.summary.verification_status == "invalidated by external file changes"
    assert any("External file change" in issue for issue in memory.working.unresolved_errors)


def test_resume_detects_external_change_after_snapshot_edit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file_path = workspace / "app.py"
    file_path.write_text("old\n", encoding="utf-8")
    store = MemoryStore(tmp_path / "session")
    memory = MemoryState(session_id=store.session_id)
    memory.start_task(
        task="task",
        workspace=str(workspace.resolve()),
        system_prompt=SYSTEM_PROMPT,
        safety_rules=[],
        completion_rules=[],
        max_steps=30,
    )
    runtime = EditingRuntime(
        WorkspacePathPolicy(workspace),
        snapshot_dir=store.session_dir / "snapshots",
    )
    read_tool = ReadFileTool(runtime.path_policy, runtime)
    edit_tool = EditFileTool(runtime)
    read = read_tool.execute({"path": "app.py", "start_line": 1, "end_line": 1})
    assert read.ok and read.metadata
    memory.record_tool_result(
        ToolCall(id="read", name="read_file", arguments={"path": "app.py"}),
        read,
        step=1,
    )
    edit_call = ToolCall(
        id="edit",
        name="edit_file",
        arguments={
            "path": "app.py",
            "snapshot_id": read.metadata["snapshot_id"],
            "operations": [{"op": "replace", "start_line": 1, "end_line": 1, "new_lines": ["new"]}],
        },
    )
    edited = edit_tool.execute(edit_call.arguments)
    assert edited.ok
    memory.record_tool_result(edit_call, edited, step=2)
    store.save_state(memory, status="paused")
    file_path.write_text("external\n", encoding="utf-8")

    changed = store.validate_workspace(memory, workspace)

    assert changed == ["app.py"]
    assert "app.py" not in memory.current_snapshots
    assert "app.py" in memory.invalidated_files


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_context_tokens": 4_000, "reserved_output_tokens": 4_000},
        {"compaction_threshold": 0.9, "hard_limit_threshold": 0.9},
    ],
)
def test_memory_config_rejects_invalid_budget(arguments: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(**arguments)  # type: ignore[arg-type]
