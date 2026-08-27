"""Persistent session events, state snapshots, command outputs, and file snapshots."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from repo_rivet.memory.models import MemoryState, add_unique
from repo_rivet.storage.event_logger import EventLogger
from repo_rivet.tools.base import ToolCall, ToolResult

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
            if call.name in {"run_command", "git_diff"}
            else self.file_snapshot_dir
        )
        directory.mkdir(parents=True, exist_ok=True)
        safe_id = _SAFE_NAME.sub("-", call.id)[:80] or "call"
        suffix = ".log" if call.name in {"run_command", "git_diff"} else ".txt"
        output_path = directory / f"step-{step:04d}-{safe_id}{suffix}"
        sanitized_output = self._event_logger.sanitize(result.raw_output)
        output_path.write_text(str(sanitized_output), encoding="utf-8")
        return output_path.relative_to(self.session_dir).as_posix()

    def save_state(self, memory: MemoryState, **runtime_state: Any) -> None:
        payload = {
            "memory": memory.model_dump(mode="json"),
            "runtime": runtime_state,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        temporary_path = self.state_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary_path.replace(self.state_path)

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
                memory.invalidated_files.add(path)
                issue = f"External file change detected; reread required: {path}"
                add_unique(memory.working.unresolved_errors, issue, limit=20)
                add_unique(memory.summary.unresolved_issues, issue)
        if changed:
            memory.last_verification_success = False
            memory.working.last_verification_result = "invalidated by external file changes"
            memory.summary.verification_status = "invalidated by external file changes"
        return changed
