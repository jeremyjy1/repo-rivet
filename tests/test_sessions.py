import json
import os
import socket
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from repo_rivet.cli import cli
from repo_rivet.llm.protocol import validate_tool_call_protocol
from repo_rivet.memory.context_manager import SYSTEM_PROMPT
from repo_rivet.memory.models import Message
from repo_rivet.session.errors import (
    AmbiguousSessionId,
    SessionAlreadyRunning,
    SessionNotResumable,
)
from repo_rivet.session.models import SessionStatus
from repo_rivet.session.store import FileSessionStore
from repo_rivet.storage.atomic_write import atomic_write_json
from repo_rivet.tools.base import ToolCall
from repo_rivet.tools.registry import create_default_registry


def make_store(tmp_path: Path) -> FileSessionStore:
    return FileSessionStore(tmp_path / "reporivet-home")


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "reporivet.toml"
    path.write_text(
        """
[api]
api_key = "test-secret"
base_url = "https://example.com/v1"
model = "test-model"
context_window_tokens = 32768
""",
        encoding="utf-8",
    )
    return path


def test_create_lists_global_session_and_sets_workspace_pointer(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    manager = make_store(tmp_path)

    loaded = manager.create(workspace=workspace, task="fix the parser", name="parser-fix")

    assert manager.list_sessions(workspace=workspace) == [loaded.metadata]
    assert manager.get_active(workspace) == loaded.metadata
    assert loaded.store.state_path.is_file()
    assert (loaded.store.session_dir / "meta.json").is_file()
    assert (loaded.store.session_dir / "summary.json").is_file()
    assert loaded.store.events_path.is_file()
    assert not (workspace / ".reporivet").exists()


def test_unique_short_id_resolves_and_ambiguous_id_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifiers = iter(
        [
            "20260828-153022-a7c4e1",
            "20260828-160102-a7c9f2",
        ]
    )
    monkeypatch.setattr("repo_rivet.session.store.create_session_id", lambda: next(identifiers))
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    first = manager.create(workspace=workspace)
    manager.create(workspace=workspace)

    assert manager.resolve_id("a7c4e1") == first.metadata.session_id
    with pytest.raises(AmbiguousSessionId, match="a7c"):
        manager.resolve_id("a7c")


def test_fork_preserves_memory_but_drops_session_approvals(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    manager = make_store(tmp_path)
    source = manager.create(workspace=workspace, task="fix auth", name="auth")
    source.memory.start_task(
        task="fix auth",
        workspace=str(workspace.resolve()),
        system_prompt=SYSTEM_PROMPT,
        safety_rules=[],
        completion_rules=[],
        max_steps=30,
    )
    source.memory.summary.completed_actions.append("located auth module")
    source.memory.approval_session_grants["grant"] = {"action": "approve"}
    source.memory.denied_request_fingerprints.add("denied")
    source.memory.approval_denial_guidance["denied"] = "use a narrower edit"
    source.memory.messages.extend(
        [
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "read-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            ),
            Message(role="tool", tool_call_id="read-1", content='{"ok": true}'),
        ]
    )
    (workspace / "app.py").write_text("old\n", encoding="utf-8")
    source_registry = create_default_registry(
        workspace,
        snapshot_dir=source.store.session_dir / "snapshots",
    )
    read = source_registry.execute(
        ToolCall(id="snapshot-read", name="read_file", arguments={"path": "app.py"})
    )
    assert read.ok and read.metadata
    source.memory.current_snapshots["app.py"] = read.metadata["snapshot_id"]
    source.store.save_state(source.memory, status=SessionStatus.PAUSED.value)

    forked = manager.fork(source.metadata.session_id, name="alternative", set_active=True)

    assert forked.metadata.parent_session_id == source.metadata.session_id
    assert forked.metadata.status == SessionStatus.CREATED
    assert forked.memory.summary.completed_actions == ["located auth module"]
    assert forked.memory.approval_session_grants == {}
    assert forked.memory.denied_request_fingerprints == set()
    assert forked.memory.approval_denial_guidance == {}
    assert manager.get_active(workspace) == forked.metadata
    assert forked.store.reconcile_interrupted_tool_calls(forked.memory) == []
    forked_registry = create_default_registry(
        workspace,
        snapshot_dir=forked.store.session_dir / "snapshots",
    )
    edited = forked_registry.execute(
        ToolCall(
            id="fork-edit",
            name="edit_file",
            arguments={
                "path": "app.py",
                "snapshot_id": read.metadata["snapshot_id"],
                "operations": [
                    {
                        "op": "replace",
                        "start_line": 1,
                        "end_line": 1,
                        "new_lines": ["forked"],
                    }
                ],
            },
        )
    )
    assert edited.ok
    assert (workspace / "app.py").read_text(encoding="utf-8") == "forked\n"


def test_completed_and_failed_sessions_must_be_forked(tmp_path: Path) -> None:
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    loaded = manager.create(workspace=workspace)

    for status in (SessionStatus.COMPLETED, SessionStatus.FAILED):
        loaded.memory.status = status.value
        loaded.store.save_state(loaded.memory, status=status.value)
        metadata = manager.read_metadata(loaded.metadata.session_id)
        with pytest.raises(SessionNotResumable, match="fork"):
            manager.ensure_resumable(metadata)


def test_archive_is_hidden_and_clears_active_pointer(tmp_path: Path) -> None:
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    loaded = manager.create(workspace=workspace)

    archived = manager.archive(loaded.metadata.session_id)

    assert archived.status == SessionStatus.ARCHIVED
    assert manager.list_sessions(workspace=workspace) == []
    assert manager.list_sessions(workspace=workspace, include_archived=True) == [archived]
    assert manager.get_active(workspace) is None


def test_delete_moves_session_to_trash_and_clears_pointer(tmp_path: Path) -> None:
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    loaded = manager.create(workspace=workspace)

    destination = manager.delete(loaded.metadata.session_id)

    assert destination.is_dir()
    assert not loaded.store.session_dir.exists()
    assert manager.get_active(workspace) is None


def test_live_session_lock_prevents_second_owner(tmp_path: Path) -> None:
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    loaded = manager.create(workspace=workspace)

    with manager.lock(loaded.metadata.session_id), pytest.raises(SessionAlreadyRunning):
        manager.lock(loaded.metadata.session_id).__enter__()


@pytest.mark.parametrize(
    "active_status",
    [
        SessionStatus.RUNNING,
        SessionStatus.VERIFYING,
        SessionStatus.AWAITING_VERIFICATION_PLAN,
    ],
)
def test_repair_removes_stale_lock_and_pauses_interrupted_session(
    tmp_path: Path,
    active_status: SessionStatus,
) -> None:
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    loaded = manager.create(workspace=workspace)
    loaded.memory.status = active_status.value
    loaded.store.save_state(loaded.memory, status=active_status.value)
    lock_path = loaded.store.session_dir / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 999_999_999,
                "hostname": socket.gethostname(),
                "created_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    repairs = manager.repair(loaded.metadata.session_id)

    assert any("stale lock" in item for item in repairs)
    assert not lock_path.exists()
    assert manager.read_metadata(loaded.metadata.session_id).status == SessionStatus.PAUSED


def test_interrupted_tool_call_is_marked_uncertain_without_retry(tmp_path: Path) -> None:
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    loaded = manager.create(workspace=workspace)
    loaded.store.log(
        "tool_call",
        tool_call_id="write-1",
        name="write_file",
        arguments={"path": "src/app.py"},
    )

    interrupted = loaded.store.reconcile_interrupted_tool_calls(loaded.memory)

    assert interrupted == ["write_file (write-1)"]
    assert any(
        "side effects may be unknown" in (message.content or "")
        for message in loaded.memory.messages
    )
    assert all(message.role != "tool" for message in loaded.memory.messages)


def test_interrupted_assistant_tool_group_receives_synthetic_results(tmp_path: Path) -> None:
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    loaded = manager.create(workspace=workspace)
    loaded.memory.messages.append(
        Message(
            role="assistant",
            tool_calls=[
                {
                    "id": "write-1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                }
            ],
        )
    )
    loaded.store.log("tool_call", tool_call_id="write-1", name="write_file")

    loaded.store.reconcile_interrupted_tool_calls(loaded.memory)

    result = loaded.memory.messages[-2]
    assert result.role == "tool"
    assert result.tool_call_id == "write-1"
    assert "not retried" in (result.content or "")
    assert loaded.memory.messages[-1].role == "system"


def test_interrupted_tool_group_is_repaired_in_place_after_following_messages(
    tmp_path: Path,
) -> None:
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    loaded = manager.create(workspace=workspace)
    loaded.memory.messages.extend(
        [
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "write-1",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            ),
            Message(role="user", content="continue after interruption"),
            Message(role="tool", tool_call_id="write-1", content='{"ok": true}'),
        ]
    )

    repairs = loaded.store.reconcile_interrupted_tool_calls(loaded.memory)

    validate_tool_call_protocol([message.as_chat_message() for message in loaded.memory.messages])
    assert repairs == ["write_file (write-1)", "orphan tool result (write-1)"]
    assert [message.role for message in loaded.memory.messages[:3]] == [
        "assistant",
        "tool",
        "user",
    ]
    assert "interrupted_tool_call" in (loaded.memory.messages[1].content or "")


def test_repair_truncates_only_invalid_final_event_and_keeps_backup(tmp_path: Path) -> None:
    manager = make_store(tmp_path)
    workspace = tmp_path / "project"
    workspace.mkdir()
    loaded = manager.create(workspace=workspace)
    with loaded.store.events_path.open("a", encoding="utf-8") as output:
        output.write('{"incomplete":')

    repairs = manager.repair(loaded.metadata.session_id)

    assert any("invalid final event" in item for item in repairs)
    assert loaded.store.events_path.with_suffix(".jsonl.corrupt").is_file()
    for line in loaded.store.events_path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_atomic_json_failure_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"version": 1}\n', encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        atomic_write_json(target, {"version": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}


def test_cli_use_only_changes_pointer_and_does_not_require_api_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.setenv("REPORIVET_HOME", str(home))
    manager = FileSessionStore()
    first = manager.create(workspace=workspace)
    second = manager.create(workspace=workspace)
    manager.set_active(workspace, first.metadata.session_id)
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    exit_code = cli(
        ["session", "use", second.metadata.short_id, "--workspace", str(workspace)],
        console=console,
    )

    assert exit_code == 0
    assert manager.get_active(workspace) == second.metadata
    output = buffer.getvalue()
    assert "selection is scoped to the session workspace" in output
    assert f"reporivet session resume {second.metadata.short_id}" in output


def test_cli_use_in_session_workspace_suggests_bare_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.setenv("REPORIVET_HOME", str(home))
    monkeypatch.chdir(workspace)
    session = FileSessionStore().create(workspace=workspace)
    buffer = StringIO()

    exit_code = cli(
        ["session", "use", session.metadata.short_id],
        console=Console(file=buffer, force_terminal=False, color_system=None),
    )

    assert exit_code == 0
    assert "Run `reporivet session resume` to continue." in buffer.getvalue()


def test_resume_without_workspace_pointer_explains_workspace_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REPORIVET_HOME", str(tmp_path / "home"))
    workspace = tmp_path / "project"
    workspace.mkdir()
    buffer = StringIO()

    exit_code = cli(
        ["session", "resume", "--workspace", str(workspace)],
        console=Console(file=buffer, force_terminal=False, color_system=None),
    )

    assert exit_code == 2
    output = " ".join(buffer.getvalue().split())
    assert "Active sessions are workspace-scoped" in output
    assert "resume by ID" in output


def test_chat_uses_active_session_then_explicit_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "project"
    workspace.mkdir()
    config = write_config(tmp_path)
    monkeypatch.setenv("REPORIVET_HOME", str(home))
    manager = FileSessionStore()
    active = manager.create(workspace=workspace, name="active")
    explicit = manager.create(workspace=workspace, name="explicit")
    manager.set_active(workspace, active.metadata.session_id)
    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    assert (
        cli(
            ["chat", "--workspace", str(workspace), "--config", str(config)],
            console=console,
            prompt_reader=lambda _: "/exit",
        )
        == 0
    )
    assert manager.get_active(workspace).session_id == active.metadata.session_id  # type: ignore[union-attr]

    assert (
        cli(
            [
                "chat",
                "--workspace",
                str(workspace),
                "--config",
                str(config),
                "--session",
                explicit.metadata.short_id,
            ],
            console=console,
            prompt_reader=lambda _: "/exit",
        )
        == 0
    )
    assert manager.get_active(workspace).session_id == explicit.metadata.session_id  # type: ignore[union-attr]


def test_session_resume_without_id_uses_current_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "project"
    workspace.mkdir()
    config = write_config(tmp_path)
    monkeypatch.setenv("REPORIVET_HOME", str(home))
    manager = FileSessionStore()
    current = manager.create(workspace=workspace)
    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    def exit_while_running(_: str) -> str:
        metadata = manager.read_metadata(current.metadata.session_id)
        assert metadata.status == SessionStatus.RUNNING
        assert (current.store.session_dir / "lock.json").is_file()
        return "/exit"

    exit_code = cli(
        [
            "session",
            "resume",
            "--workspace",
            str(workspace),
            "--config",
            str(config),
        ],
        console=console,
        prompt_reader=exit_while_running,
    )

    assert exit_code == 0
    assert manager.read_metadata(current.metadata.session_id).status == SessionStatus.PAUSED
    assert not (current.store.session_dir / "lock.json").exists()
