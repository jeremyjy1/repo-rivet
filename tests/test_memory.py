import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_rivet.memory.context_manager import SYSTEM_PROMPT
from repo_rivet.memory.models import MemoryConfig, MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.tools.base import ToolCall, ToolResult


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


def test_repeated_unchanged_file_read_does_not_repeat_content() -> None:
    memory = make_memory()

    memory.record_tool_result(read_call(), read_result("large file content"), step=1)
    memory.record_tool_result(read_call(), read_result("large file content"), step=2)

    latest_message = json.loads(memory.messages[-1].content or "{}")
    assert "is unchanged" in latest_message["output"]
    assert "large file content" not in latest_message["output"]
    assert memory.file_memories["src/app.py"].last_read_step == 2


def test_file_modification_invalidates_previous_file_memory() -> None:
    memory = make_memory()
    memory.record_tool_result(read_call(), read_result("old content"), step=1)
    write = ToolCall(
        id="write-1",
        name="replace_text",
        arguments={"path": "src/app.py", "old_text": "old", "new_text": "new"},
    )

    memory.record_tool_result(
        write,
        ToolResult(ok=True, output="replaced", metadata={"sha256": "new-hash"}),
        step=2,
    )

    assert "src/app.py" not in memory.file_memories
    assert "src/app.py" in memory.invalidated_files
    assert not memory.last_verification_success
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
    call = ToolCall(id="command-1", name="run_command", arguments={"command": "pytest -q"})
    result = ToolResult(
        ok=True,
        output="Exit code: 1\nFAILED tests/test_app.py",
        metadata={"exit_code": 1, "timed_out": False},
        raw_output="Exit code: 1\nFAILED tests/test_app.py\n1 failed",
    )

    memory.record_tool_result(
        call,
        result,
        step=3,
        full_output_path="command_outputs/step-3.log",
    )

    assert memory.summary.verification_status == "failed: pytest -q"
    assert any("Verification failed" in issue for issue in memory.summary.unresolved_issues)
    assert memory.command_outputs[-1].full_output_path == "command_outputs/step-3.log"
    assert memory.command_outputs[-1].context_output.endswith(
        "Full output: command_outputs/step-3.log"
    )
    assert memory.command_outputs[-1].original_chars == len(result.raw_output or "")
    assert memory.command_outputs[-1].estimated_tokens > 0


def test_new_modification_invalidates_passed_verification() -> None:
    memory = make_memory()
    command = ToolCall(id="command-1", name="run_command", arguments={"command": "pytest -q"})
    memory.record_tool_result(
        command,
        ToolResult(ok=True, output="1 passed", metadata={"exit_code": 0}),
        step=1,
    )
    assert memory.last_verification_success

    write = ToolCall(id="write-1", name="write_file", arguments={"path": "new.py"})
    memory.record_tool_result(write, ToolResult(ok=True, output="written"), step=2)

    assert not memory.last_verification_success
    assert memory.last_file_change_step == 2


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
    memory.last_verification_success = True
    memory.summary.verification_status = "passed: pytest -q"
    file_path.write_text("changed externally", encoding="utf-8")

    changed = store.validate_workspace(memory, workspace)

    assert changed == ["src/app.py"]
    assert "src/app.py" in memory.invalidated_files
    assert not memory.last_verification_success
    assert memory.summary.verification_status == "invalidated by external file changes"
    assert any("External file change" in issue for issue in memory.working.unresolved_errors)


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
