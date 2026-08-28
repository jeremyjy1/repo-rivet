"""Snapshot capture, edit preflight, approval payloads, and atomic commit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from repo_rivet.editing.atomic_writer import atomic_replace_bytes
from repo_rivet.editing.document import TextDocument
from repo_rivet.editing.models import EditError, EditFileArguments, EditResult, FileSnapshot
from repo_rivet.editing.planner import PlannedEdit, plan_edit
from repo_rivet.editing.snapshot_store import SnapshotStore, VisibilityStore
from repo_rivet.safety.path_policy import WorkspacePathPolicy


class EventSink(Protocol):
    def log(self, event_type: str, **data: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedEdit:
    key: str
    path: Path
    relative_path: str
    expected_live_hash: str
    old_snapshot: FileSnapshot
    planned: PlannedEdit


class EditingRuntime:
    """Own session-local snapshots, visibility, and prepared single-file edits."""

    def __init__(
        self,
        path_policy: WorkspacePathPolicy,
        *,
        snapshot_dir: Path | None = None,
        event_logger: EventSink | None = None,
        initial_workspace_revision: int = 0,
    ) -> None:
        self.path_policy = path_policy
        self.snapshots = SnapshotStore(snapshot_dir)
        self.visibility = VisibilityStore(snapshot_dir)
        self.event_logger = event_logger
        self.workspace_revision = initial_workspace_revision
        self._prepared: dict[str, PreparedEdit] = {}
        self._noop_path = snapshot_dir / "noop-counts.json" if snapshot_dir is not None else None
        self._noop_counts = self._load_noop_counts()

    def capture(
        self,
        path: str,
        *,
        start_line: int,
        end_line: int,
        source: Literal["read_file", "search_text", "edit_file"],
        parent_snapshot_id: str | None = None,
    ) -> tuple[TextDocument, FileSnapshot]:
        resolved = self.path_policy.resolve(path)
        document = TextDocument.load(resolved)
        snapshot = self.capture_document(
            path,
            document,
            start_line=start_line,
            end_line=end_line,
            source=source,
            parent_snapshot_id=parent_snapshot_id,
        )
        return document, snapshot

    def capture_document(
        self,
        path: str,
        document: TextDocument,
        *,
        start_line: int,
        end_line: int,
        source: Literal["read_file", "search_text", "edit_file"],
        parent_snapshot_id: str | None = None,
        visible: bool = True,
    ) -> FileSnapshot:
        resolved = self.path_policy.resolve(path)
        relative = resolved.relative_to(self.path_policy.workspace).as_posix()
        snapshot = self.snapshots.put(
            document.to_snapshot(
                relative_path=relative,
                parent_snapshot_id=parent_snapshot_id,
            )
        )
        if visible:
            self.visibility.record(
                path=relative,
                snapshot_id=snapshot.snapshot_id,
                start_line=max(1, start_line),
                end_line=max(1, end_line),
                source=source,
            )
        self._log(
            "snapshot_created",
            path=relative,
            snapshot_id=snapshot.snapshot_id,
            snapshot_tag=snapshot.display_tag,
            parent_snapshot_id=parent_snapshot_id,
        )
        return snapshot

    def prepare(self, arguments: EditFileArguments) -> PreparedEdit:
        key = self._request_key(arguments)
        cached = self._prepared.get(key)
        if cached is not None:
            return cached
        resolved = self.path_policy.resolve(arguments.path)
        relative = resolved.relative_to(self.path_policy.workspace).as_posix()
        snapshot = self.snapshots.get(arguments.snapshot_id)
        if snapshot.relative_path != relative:
            raise EditError(
                "snapshot_path_mismatch",
                f"Snapshot {snapshot.display_tag} belongs to "
                f"{snapshot.relative_path}, not {relative}",
                retryable=False,
            )
        live = TextDocument.load(resolved)
        if live.raw_hash != snapshot.raw_bytes_hash:
            current = self.snapshots.put(live.to_snapshot(relative_path=relative))
            raise EditError(
                "stale_snapshot",
                "File changed after it was read; read it again before editing",
                metadata={
                    "requested_snapshot": snapshot.display_tag,
                    "current_snapshot": current.display_tag,
                },
            )
        planned = plan_edit(snapshot, arguments, self.visibility)
        if planned.desired_document.raw_bytes == live.raw_bytes:
            self._record_noop(key)
        prepared = PreparedEdit(
            key=key,
            path=resolved,
            relative_path=relative,
            expected_live_hash=live.raw_hash,
            old_snapshot=snapshot,
            planned=planned,
        )
        self._prepared[key] = prepared
        self._log(
            "edit_prepared",
            path=relative,
            snapshot_id=snapshot.snapshot_id,
            snapshot_tag=snapshot.display_tag,
            operation_count=len(arguments.operations),
            prepared_live_hash=live.raw_hash,
        )
        return prepared

    def approval_arguments(self, arguments: EditFileArguments) -> dict[str, Any]:
        prepared = self.prepare(arguments)
        return {
            "path": prepared.relative_path,
            "snapshot_id": prepared.old_snapshot.snapshot_id,
            "snapshot_tag": prepared.old_snapshot.display_tag,
            "operations": [self._operation_summary(item) for item in arguments.operations],
            "prepared_live_hash": prepared.expected_live_hash,
            "diff_preview": prepared.planned.diff_preview,
        }

    def record_approval(self, arguments: EditFileArguments, *, source: str) -> None:
        prepared = self.prepare(arguments)
        self._log(
            "edit_approved",
            path=prepared.relative_path,
            snapshot_id=prepared.old_snapshot.snapshot_id,
            snapshot_tag=prepared.old_snapshot.display_tag,
            source=source,
        )

    def commit(self, arguments: EditFileArguments) -> EditResult:
        prepared = self.prepare(arguments)
        latest = TextDocument.load(prepared.path)
        if latest.raw_hash != prepared.expected_live_hash:
            current = self.snapshots.put(latest.to_snapshot(relative_path=prepared.relative_path))
            raise EditError(
                "edit_changed_during_approval",
                "File changed during edit preflight or approval; no changes were written",
                metadata={
                    "requested_snapshot": prepared.old_snapshot.display_tag,
                    "current_snapshot": current.display_tag,
                },
            )
        atomic_replace_bytes(prepared.path, prepared.planned.desired_document.raw_bytes)
        self.workspace_revision += 1
        new_snapshot = self.snapshots.put(
            prepared.planned.desired_document.to_snapshot(
                relative_path=prepared.relative_path,
                parent_snapshot_id=prepared.old_snapshot.snapshot_id,
            )
        )
        for start_line, end_line in prepared.planned.changed_ranges:
            self.visibility.record(
                path=prepared.relative_path,
                snapshot_id=new_snapshot.snapshot_id,
                start_line=start_line,
                end_line=end_line,
                source="edit_file",
            )
        self._prepared.pop(prepared.key, None)
        self._log(
            "edit_committed",
            path=prepared.relative_path,
            old_snapshot_id=prepared.old_snapshot.snapshot_id,
            old_snapshot_tag=prepared.old_snapshot.display_tag,
            new_snapshot_id=new_snapshot.snapshot_id,
            new_snapshot_tag=new_snapshot.display_tag,
        )
        return EditResult(
            path=prepared.relative_path,
            old_snapshot_id=prepared.old_snapshot.snapshot_id,
            old_snapshot_tag=prepared.old_snapshot.display_tag,
            new_snapshot_id=new_snapshot.snapshot_id,
            new_snapshot_tag=new_snapshot.display_tag,
            changed_ranges=prepared.planned.changed_ranges,
            bytes_before=len(latest.raw_bytes),
            bytes_after=len(prepared.planned.desired_document.raw_bytes),
            workspace_revision=self.workspace_revision,
            diff_preview=prepared.planned.diff_preview,
        )

    def record_created_file(self) -> int:
        """Advance the shared workspace revision after a successful create."""
        self.workspace_revision += 1
        return self.workspace_revision

    def _record_noop(self, key: str) -> None:
        count = self._noop_counts.get(key, 0) + 1
        self._noop_counts[key] = count
        self._save_noop_counts()
        if count >= 3:
            raise EditError(
                "edit_loop_detected",
                "The same no-op edit was submitted three times; reread before editing again",
                retryable=False,
            )
        message = "Edit result is byte-identical to the current file"
        if count == 2:
            message += "; reread the file before retrying"
        raise EditError("edit_noop", message)

    def _load_noop_counts(self) -> dict[str, int]:
        if self._noop_path is None or not self._noop_path.exists():
            return {}
        try:
            payload = json.loads(self._noop_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            key: value
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, int) and value > 0
        }

    def _save_noop_counts(self) -> None:
        if self._noop_path is None:
            return
        self._noop_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_bytes(
            self._noop_path,
            (json.dumps(self._noop_counts, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )

    @staticmethod
    def _request_key(arguments: EditFileArguments) -> str:
        payload = json.dumps(
            arguments.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _operation_summary(operation: Any) -> dict[str, Any]:
        payload = operation.model_dump(mode="json")
        new_lines = payload.pop("new_lines", None)
        if isinstance(new_lines, list):
            serialized = json.dumps(new_lines, ensure_ascii=False, separators=(",", ":"))
            payload["new_line_count"] = len(new_lines)
            payload["new_lines_sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return payload

    def _log(self, event_type: str, **data: Any) -> None:
        if self.event_logger is not None:
            self.event_logger.log(event_type, **data)
