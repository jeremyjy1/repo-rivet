"""Filesystem implementation of global multi-session management."""

import copy
import hashlib
import json
import os
import shutil
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from repo_rivet.memory.models import MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.session.errors import (
    AmbiguousSessionId,
    SessionCorrupted,
    SessionError,
    SessionLockStale,
    SessionNotFound,
    SessionNotResumable,
    SessionWorkspaceMismatch,
)
from repo_rivet.session.lock import SessionLock, process_is_alive
from repo_rivet.session.models import ActiveSessionPointer, SessionMetadata, SessionStatus
from repo_rivet.storage.atomic_write import atomic_write_json


def get_reporivet_home() -> Path:
    """Resolve the global data root without writing into a target repository."""
    configured = os.environ.get("REPORIVET_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".reporivet"


def workspace_key(workspace: str | Path) -> str:
    normalized = Path(workspace).expanduser().resolve()
    return hashlib.sha256(str(normalized).encode()).hexdigest()


def create_session_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:6]}"


@dataclass(slots=True)
class LoadedSession:
    metadata: SessionMetadata
    memory: MemoryState
    store: MemoryStore


class FileSessionStore:
    """Manage metadata, active pointers, lifecycle operations, and session locks."""

    def __init__(self, root: str | Path | None = None, *, secrets: tuple[str, ...] = ()) -> None:
        self.root = Path(root) if root is not None else get_reporivet_home()
        self.root = self.root.expanduser().resolve()
        self.sessions_dir = self.root / "sessions"
        self.workspaces_dir = self.root / "workspaces"
        self.trash_dir = self.root / "trash"
        self.secrets = secrets

    def create(
        self,
        *,
        workspace: str | Path,
        task: str = "",
        name: str | None = None,
        model: str | None = None,
        parent_session_id: str | None = None,
        memory: MemoryState | None = None,
        set_active: bool = True,
    ) -> LoadedSession:
        normalized_workspace = Path(workspace).expanduser().resolve()
        session_id = create_session_id()
        session_dir = self.sessions_dir / session_id
        store = MemoryStore(session_dir, secrets=self.secrets)
        state = memory or MemoryState(session_id=session_id)
        state.session_id = session_id
        for event in state.reasoning_events:
            event.session_id = session_id
        for event in state.observation_events:
            event.session_id = session_id
        state.status = SessionStatus.CREATED.value
        now = datetime.now(UTC)
        preview = " ".join(task.strip().split())[:160]
        metadata = SessionMetadata(
            session_id=session_id,
            name=self._normalize_name(name, preview, session_id),
            task_preview=preview,
            workspace=str(normalized_workspace),
            workspace_key=workspace_key(normalized_workspace),
            status=SessionStatus.CREATED,
            created_at=now,
            updated_at=now,
            model=model,
            parent_session_id=parent_session_id,
        )
        atomic_write_json(session_dir / "meta.json", metadata.model_dump(mode="json"))
        store.save_state(state, status=SessionStatus.CREATED.value, agent_step=0)
        store.log("session_created", parent_session_id=parent_session_id)
        if set_active:
            self.set_active(normalized_workspace, session_id)
        return LoadedSession(metadata=self.read_metadata(session_id), memory=state, store=store)

    def resolve_id(self, reference: str) -> str:
        normalized = reference.strip()
        if not normalized:
            raise SessionNotFound("Session ID must not be empty")
        exact = self.sessions_dir / normalized
        if (exact / "meta.json").is_file():
            return normalized
        matches = [
            path.name
            for path in self.sessions_dir.glob("*")
            if (path / "meta.json").is_file()
            and (
                path.name.startswith(normalized)
                or path.name.endswith(normalized)
                or path.name.rsplit("-", maxsplit=1)[-1].startswith(normalized)
            )
        ]
        if not matches:
            raise SessionNotFound(f"Session not found: {reference}")
        if len(matches) > 1:
            rendered = "\n".join(f"- {item}" for item in sorted(matches))
            raise AmbiguousSessionId(
                f"Session '{reference}' is ambiguous. Matches:\n{rendered}\nProvide a longer ID."
            )
        return matches[0]

    def read_metadata(self, reference: str) -> SessionMetadata:
        session_id = self.resolve_id(reference)
        path = self.sessions_dir / session_id / "meta.json"
        try:
            return SessionMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as error:
            raise SessionCorrupted(f"Invalid session metadata {path}: {error}") from None

    def load(self, reference: str) -> LoadedSession:
        metadata = self.read_metadata(reference)
        store = MemoryStore(self.sessions_dir / metadata.session_id, secrets=self.secrets)
        try:
            memory = store.load_state()
        except ValueError as error:
            raise SessionCorrupted(str(error)) from None
        if memory.session_id != metadata.session_id:
            raise SessionCorrupted(
                f"Session ID mismatch in {store.state_path}: {memory.session_id}"
            )
        return LoadedSession(metadata=metadata, memory=memory, store=store)

    def list_sessions(
        self,
        *,
        workspace: str | Path | None = None,
        status: SessionStatus | None = None,
        include_archived: bool = False,
        limit: int = 20,
    ) -> list[SessionMetadata]:
        normalized_workspace = (
            str(Path(workspace).expanduser().resolve()) if workspace is not None else None
        )
        sessions: list[SessionMetadata] = []
        if not self.sessions_dir.exists():
            return sessions
        for path in self.sessions_dir.iterdir():
            if not (path / "meta.json").is_file():
                continue
            metadata = self.read_metadata(path.name)
            if normalized_workspace is not None and metadata.workspace != normalized_workspace:
                continue
            if status is not None and metadata.status != status:
                continue
            if not include_archived and metadata.status == SessionStatus.ARCHIVED:
                continue
            sessions.append(metadata)
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions[:limit]

    def set_active(self, workspace: str | Path, reference: str) -> SessionMetadata:
        metadata = self.read_metadata(reference)
        normalized_workspace = Path(workspace).expanduser().resolve()
        if str(normalized_workspace) != metadata.workspace:
            raise SessionWorkspaceMismatch(
                f"Session {metadata.session_id} belongs to {metadata.workspace}, "
                f"not {normalized_workspace}"
            )
        pointer = ActiveSessionPointer(
            workspace=str(normalized_workspace),
            session_id=metadata.session_id,
            updated_at=datetime.now(UTC),
        )
        atomic_write_json(
            self._active_path(normalized_workspace),
            pointer.model_dump(mode="json"),
        )
        return metadata

    def get_active(self, workspace: str | Path) -> SessionMetadata | None:
        path = self._active_path(workspace)
        if not path.exists():
            return None
        try:
            pointer = ActiveSessionPointer.model_validate_json(path.read_text(encoding="utf-8"))
            metadata = self.read_metadata(pointer.session_id)
        except (OSError, ValidationError, SessionError, ValueError) as error:
            raise SessionCorrupted(f"Invalid active session pointer {path}: {error}") from None
        normalized_workspace = str(Path(workspace).expanduser().resolve())
        if pointer.workspace != normalized_workspace or metadata.workspace != normalized_workspace:
            raise SessionCorrupted(f"Active session pointer does not match workspace: {path}")
        return metadata

    def clear_active(self, workspace: str | Path) -> None:
        self._active_path(workspace).unlink(missing_ok=True)

    def rename(self, reference: str, name: str) -> SessionMetadata:
        metadata = self.read_metadata(reference)
        normalized = " ".join(name.strip().split())
        if not normalized:
            raise ValueError("Session name must not be empty")
        metadata.name = normalized[:100]
        metadata.updated_at = datetime.now(UTC)
        self._write_metadata(metadata)
        return metadata

    def archive(self, reference: str) -> SessionMetadata:
        metadata = self.read_metadata(reference)
        self._ensure_unlocked(metadata.session_id)
        if metadata.status == SessionStatus.RUNNING:
            raise SessionNotResumable("A running session cannot be archived")
        metadata.status = SessionStatus.ARCHIVED
        metadata.updated_at = datetime.now(UTC)
        self._write_metadata(metadata)
        active = self.get_active(metadata.workspace)
        if active and active.session_id == metadata.session_id:
            self.clear_active(metadata.workspace)
        return metadata

    def fork(
        self,
        reference: str,
        *,
        name: str | None = None,
        set_active: bool = False,
    ) -> LoadedSession:
        source = self.load(reference)
        state = copy.deepcopy(source.memory)
        state.approval_session_grants.clear()
        state.denied_request_fingerprints.clear()
        state.approval_mode_override = None
        state.status = SessionStatus.CREATED.value
        for output in state.command_outputs:
            output.full_output_path = None
        task = state.fixed.original_task if state.fixed else source.metadata.task_preview
        forked = self.create(
            workspace=source.metadata.workspace,
            task=task,
            name=name or f"{source.metadata.name}-fork",
            model=source.metadata.model,
            parent_session_id=source.metadata.session_id,
            memory=state,
            set_active=set_active,
        )
        forked.store.log("session_forked", parent_session_id=source.metadata.session_id)
        return forked

    def delete(self, reference: str) -> Path:
        metadata = self.read_metadata(reference)
        self._ensure_unlocked(metadata.session_id)
        destination = self.trash_dir / metadata.session_id
        if destination.exists():
            destination = self.trash_dir / f"{metadata.session_id}-{uuid4().hex[:6]}"
        self.trash_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.sessions_dir / metadata.session_id), destination)
        self._clear_pointers_to(metadata.session_id)
        return destination

    def ensure_resumable(self, metadata: SessionMetadata) -> None:
        if metadata.status == SessionStatus.ARCHIVED:
            raise SessionNotResumable("Archived sessions cannot be resumed; fork it first")
        if metadata.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
            raise SessionNotResumable(
                f"{metadata.status.value.capitalize()} sessions cannot be resumed; fork it first"
            )

    def lock(self, reference: str) -> SessionLock:
        session_id = self.resolve_id(reference)
        return SessionLock(self.sessions_dir / session_id / "lock.json")

    def is_interrupted(self, metadata: SessionMetadata) -> bool:
        if metadata.status != SessionStatus.RUNNING:
            return False
        lock_path = self.sessions_dir / metadata.session_id / "lock.json"
        if not lock_path.exists():
            return True
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            return payload.get("hostname") == socket.gethostname() and not process_is_alive(
                int(payload.get("pid", -1))
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return True

    def repair(self, reference: str) -> list[str]:
        loaded = self.load(reference)
        repairs: list[str] = []
        lock_path = loaded.store.session_dir / "lock.json"
        if lock_path.exists():
            try:
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                hostname = str(payload["hostname"])
                pid = int(payload["pid"])
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
                raise SessionCorrupted(f"Cannot safely inspect lock {lock_path}: {error}") from None
            if hostname != socket.gethostname() or process_is_alive(pid):
                raise SessionLockStale(
                    f"Lock belongs to a possibly live process: {hostname} process {pid}"
                )
            lock_path.unlink()
            repairs.append(f"removed stale lock from process {pid}")

        if loaded.metadata.status == SessionStatus.RUNNING:
            loaded.memory.status = SessionStatus.PAUSED.value
            loaded.store.save_state(loaded.memory, status=SessionStatus.PAUSED.value)
            repairs.append("changed interrupted status from running to paused")

        repairs.extend(self._repair_event_tail(loaded.store.events_path))
        return repairs

    def _active_path(self, workspace: str | Path) -> Path:
        return self.workspaces_dir / workspace_key(workspace) / "active.json"

    def _write_metadata(self, metadata: SessionMetadata) -> None:
        atomic_write_json(
            self.sessions_dir / metadata.session_id / "meta.json",
            metadata.model_dump(mode="json"),
        )

    def _ensure_unlocked(self, session_id: str) -> None:
        lock_path = self.sessions_dir / session_id / "lock.json"
        if lock_path.exists():
            raise SessionNotResumable(
                f"Session {session_id} has a lock; repair a stale lock before changing it"
            )

    def _clear_pointers_to(self, session_id: str) -> None:
        if not self.workspaces_dir.exists():
            return
        for pointer_path in self.workspaces_dir.glob("*/active.json"):
            try:
                pointer = ActiveSessionPointer.model_validate_json(
                    pointer_path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError):
                continue
            if pointer.session_id == session_id:
                pointer_path.unlink(missing_ok=True)

    @staticmethod
    def _normalize_name(name: str | None, preview: str, session_id: str) -> str:
        candidate = " ".join((name or preview or "untitled").strip().split())
        return (candidate or session_id)[:100]

    @staticmethod
    def _repair_event_tail(path: Path) -> list[str]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        valid_count = len(lines)
        for index, line in enumerate(lines):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise SessionCorrupted(
                        f"Events are corrupted before the final line: {path}:{index + 1}"
                    ) from None
                valid_count = index
        if valid_count == len(lines):
            return []
        backup = path.with_suffix(".jsonl.corrupt")
        shutil.copy2(path, backup)
        path.write_text("".join(lines[:valid_count]), encoding="utf-8")
        return [f"removed an invalid final event line (backup: {backup.name})"]
