"""Persistent session events, state snapshots, command outputs, and file snapshots."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from repo_rivet.editing.snapshot_store import SnapshotStore
from repo_rivet.memory.models import MemoryState, Message, add_unique
from repo_rivet.storage.atomic_write import atomic_write_json
from repo_rivet.storage.event_logger import EventLogger
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.verification.models import VerificationStatus

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class MemoryStore:
    """Store immutable events and an atomic current-state snapshot for one session."""

    def __init__(self, session_dir: str | Path, *, secrets: tuple[str, ...] = ()) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.session_dir / "events.jsonl"
        self.state_path = self.session_dir / "state.json"
        self.command_output_dir = self.session_dir / "command_outputs"
        self.file_snapshot_dir = self.session_dir / "file_snapshots"
        self._event_logger = EventLogger(self.events_path, secrets=secrets)

    @classmethod
    def create(cls, base_dir: str | Path, *, secrets: tuple[str, ...] = ()) -> "MemoryStore":
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{timestamp}-{uuid4().hex[:8]}"
        return cls(Path(base_dir) / session_id, secrets=secrets)

    @property
    def session_id(self) -> str:
        return self.session_dir.name

    def log(self, event_type: str, **data: Any) -> None:
        self._event_logger.log(event_type, **data)

    def save_tool_output(self, call: ToolCall, result: ToolResult, *, step: int) -> str | None:
        """Persist a complete output away from model context and return a relative reference."""
        if result.raw_output is None:
            return None
        directory = (
            self.command_output_dir
            if call.name in {"run_command", "run_verification", "git_diff"}
            else self.file_snapshot_dir
        )
        directory.mkdir(parents=True, exist_ok=True)
        safe_id = _SAFE_NAME.sub("-", call.id)[:80] or "call"
        suffix = ".log" if call.name in {"run_command", "run_verification", "git_diff"} else ".txt"
        output_path = directory / f"step-{step:04d}-{safe_id}{suffix}"
        sanitized_output = self._event_logger.sanitize(result.raw_output)
        output_path.write_text(str(sanitized_output), encoding="utf-8")
        return output_path.relative_to(self.session_dir).as_posix()

    def save_state(self, memory: MemoryState, **runtime_state: Any) -> None:
        status = str(runtime_state.get("status", memory.status))
        normalized_status = {
            "ready": "created",
            "stopped": "paused",
            "error": "failed",
        }.get(status, status)
        if "status" in runtime_state:
            runtime_state["status"] = normalized_status
            memory.status = normalized_status
        memory_payload = memory.model_dump(mode="json")
        memory_payload["messages"] = [
            message.model_dump(mode="json")
            for message in memory.messages
            if not message.ephemeral and message.is_valid_durable_message()
        ]
        payload = {
            "memory": memory_payload,
            "runtime": runtime_state,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(self.state_path, payload)
        atomic_write_json(
            self.session_dir / "summary.json",
            memory.summary.model_dump(mode="json"),
        )
        self._update_metadata(memory, runtime_state)

    def load_state(self) -> MemoryState:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError(f"Session state does not exist: {self.state_path}") from None
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load session state: {self.state_path}: {error}") from None
        try:
            return MemoryState.model_validate(payload["memory"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid session state: {self.state_path}: {error}") from None

    def validate_workspace(self, memory: MemoryState, workspace: str | Path) -> list[str]:
        """Invalidate file memories changed outside RepoRivet since the last saved state."""
        if memory.fixed is None:
            return []
        expected_workspace = Path(memory.fixed.workspace).resolve()
        actual_workspace = Path(workspace).resolve()
        if expected_workspace != actual_workspace:
            raise ValueError(
                f"Session workspace mismatch: expected {expected_workspace}, got {actual_workspace}"
            )

        changed: list[str] = []
        file_memory_paths = set(memory.file_memories)
        for path, file_memory in list(memory.file_memories.items()):
            full_path = (actual_workspace / path).resolve()
            if not full_path.is_relative_to(actual_workspace):
                current_hash = "invalid-path"
            else:
                try:
                    current_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
                except OSError:
                    current_hash = "missing"
            if current_hash != file_memory.sha256:
                changed.append(path)
                memory.file_memories.pop(path, None)
                memory.current_snapshots.pop(path, None)
                memory.invalidated_files.add(path)
                issue = f"External file change detected; reread required: {path}"
                add_unique(memory.working.unresolved_errors, issue, limit=20)
                add_unique(memory.summary.unresolved_issues, issue)
        snapshot_store = SnapshotStore(self.session_dir / "snapshots")
        for path, snapshot_id in list(memory.current_snapshots.items()):
            if path in file_memory_paths:
                continue
            full_path = (actual_workspace / path).resolve()
            if not full_path.is_relative_to(actual_workspace):
                current_hash = "missing-or-invalid"
                snapshot = None
            else:
                try:
                    snapshot = snapshot_store.get(snapshot_id)
                    current_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
                except (OSError, ValueError):
                    current_hash = "missing-or-invalid"
                    snapshot = None
            if (
                snapshot is None
                or snapshot.relative_path != path
                or current_hash != snapshot.raw_bytes_hash
            ):
                changed.append(path)
                memory.current_snapshots.pop(path, None)
                memory.invalidated_files.add(path)
                issue = f"External file change detected; reread required: {path}"
                add_unique(memory.working.unresolved_errors, issue, limit=20)
                add_unique(memory.summary.unresolved_issues, issue)
        if changed:
            memory.context_checkpoint = None
            memory.workspace_revision += 1
            if memory.runtime_v2 is not None:
                memory.runtime_v2.revisions.workspace = memory.workspace_revision
                memory.runtime_v2.revisions.knowledge += 1
            for check_id, result in list(memory.verification_results.items()):
                if result.status != VerificationStatus.STALE:
                    memory.verification_results[check_id] = result.model_copy(
                        update={"status": VerificationStatus.STALE}
                    )
            memory.working.last_verification_result = "invalidated by external file changes"
            memory.summary.verification_status = "invalidated by external file changes"
        return changed

    def reconcile_interrupted_tool_calls(self, memory: MemoryState) -> list[str]:
        """Mark tool calls with no recorded result as uncertain; never retry them implicitly."""
        started: dict[str, str] = {}
        event_started: set[str] = set()
        event_finished: set[str] = set()
        for message in memory.messages:
            if message.role == "assistant" and message.tool_calls:
                for call in message.tool_calls:
                    call_id = call.get("id")
                    if not isinstance(call_id, str):
                        continue
                    function = call.get("function", {})
                    name = function.get("name") if isinstance(function, dict) else None
                    started.setdefault(call_id, str(name or "unknown tool"))

        if self.events_path.exists():
            try:
                lines = self.events_path.read_text(encoding="utf-8").splitlines()
                for line_number, line in enumerate(lines, start=1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        if line_number == len(lines):
                            break
                        raise ValueError(
                            f"Corrupted session event log at {self.events_path}:{line_number}"
                        ) from None
                    data = event.get("data", {})
                    call_id = data.get("tool_call_id")
                    if not isinstance(call_id, str):
                        continue
                    if event.get("event") == "tool_call":
                        started[call_id] = str(data.get("name", "unknown tool"))
                        event_started.add(call_id)
                    elif event.get("event") == "tool_result":
                        event_finished.add(call_id)
            except OSError as error:
                raise ValueError(f"Could not inspect session events: {error}") from None

        def interruption_error(call_id: str, name: str) -> str:
            if name == "record_decision":
                return (
                    "Decision metadata may already be checkpointed; this meta tool has no local "
                    "side effects and was not repeated."
                )
            if call_id in event_finished:
                return "Tool completed, but its result was not checkpointed before interruption."
            if call_id in event_started:
                return (
                    "Tool call was interrupted before a result was checkpointed; side effects "
                    "are unknown and it was not retried."
                )
            return "Tool call was not started before interruption and was not retried."

        runtime_results = {
            action.tool_call_id: action.result
            for action in (
                memory.runtime_v2.actions.values() if memory.runtime_v2 is not None else ()
            )
            if action.result is not None and not action.result_applied
        }

        def persisted_result(call_id: str, _name: str) -> Message | None:
            result = runtime_results.get(call_id)
            if result is None:
                return None
            return Message.from_chat_message(
                result.to_result().as_tool_message(call_id),
                step=memory.tool_event_step,
            )

        missing, orphan_results = memory.repair_interrupted_tool_history(
            error_for=interruption_error,
            result_for=persisted_result,
        )
        for call_id, name in missing:
            started.setdefault(call_id, name)
        missing_results = [call_id for call_id, _ in missing]
        uncertain = [
            call_id
            for call_id in event_started
            if call_id not in event_finished
            and call_id not in runtime_results
            and started.get(call_id) != "record_decision"
        ]
        if not missing_results and not uncertain and not orphan_results:
            return []
        affected = list(dict.fromkeys([*missing_results, *uncertain]))
        descriptions = [f"{started[call_id]} ({call_id})" for call_id in affected]
        descriptions.extend(f"orphan tool result ({call_id})" for call_id in orphan_results)
        possible_side_effects = any(
            started.get(call_id) != "record_decision" for call_id in affected
        )
        if possible_side_effects:
            suffix = (
                ". Some side effects may be unknown. Inspect current workspace state and request "
                "approval again before repeating any write or command; nothing was retried "
                "automatically."
            )
        else:
            suffix = ". The interrupted meta tool had no local side effects and was not repeated."
        warning = (
            "The previous process stopped with incomplete tool-call checkpoints: "
            + ", ".join(descriptions)
            + suffix
        )
        already_recorded = any(
            message.role == "system" and message.content == warning for message in memory.messages
        )
        if already_recorded and not missing and not orphan_results:
            return []
        if not already_recorded:
            memory.messages.append(Message(role="system", content=warning))
        add_unique(memory.working.unresolved_errors, warning, limit=20)
        add_unique(memory.summary.unresolved_issues, warning)
        return descriptions

    def _update_metadata(self, memory: MemoryState, runtime_state: dict[str, Any]) -> None:
        metadata_path = self.session_dir / "meta.json"
        if not metadata_path.exists():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Could not update session metadata: {metadata_path}: {error}"
            ) from None
        metadata["status"] = str(runtime_state.get("status", memory.status))
        metadata["updated_at"] = datetime.now(UTC).isoformat()
        step = runtime_state.get("agent_step")
        if isinstance(step, int):
            metadata["step"] = step
        if memory.fixed is not None:
            metadata["task_preview"] = " ".join(memory.fixed.original_task.split())[:160]
        atomic_write_json(metadata_path, metadata)
