"""Content-addressed snapshots and persisted model-visible line ranges."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Literal

from repo_rivet.editing.atomic_writer import atomic_replace_bytes
from repo_rivet.editing.models import EditError, FileSnapshot, VisibleRange


class SnapshotStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self._snapshots: dict[str, FileSnapshot] = {}

    def put(self, snapshot: FileSnapshot) -> FileSnapshot:
        existing = self._snapshots.get(snapshot.snapshot_id)
        if existing is not None:
            return existing
        if self.root is not None:
            path = self.root / f"{snapshot.snapshot_id}.json.gz"
            if path.exists():
                existing = self.get(snapshot.snapshot_id)
                return existing
            payload = json.dumps(
                snapshot.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_replace_bytes(path, gzip.compress(payload))
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def get(self, snapshot_id: str) -> FileSnapshot:
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is not None:
            return snapshot
        if self.root is None:
            raise EditError("snapshot_not_found", f"Unknown snapshot: {snapshot_id[:8]}")
        path = self.root / f"{snapshot_id}.json.gz"
        try:
            payload = json.loads(gzip.decompress(path.read_bytes()))
            snapshot = FileSnapshot.model_validate(payload)
        except FileNotFoundError:
            raise EditError("snapshot_not_found", f"Unknown snapshot: {snapshot_id[:8]}") from None
        except (OSError, ValueError, json.JSONDecodeError):
            raise EditError(
                "snapshot_corrupted",
                f"Snapshot is corrupted: {snapshot_id[:8]}",
                retryable=False,
            ) from None
        self._snapshots[snapshot_id] = snapshot
        return snapshot


class VisibilityStore:
    def __init__(self, root: Path | None = None) -> None:
        self.path = root / "visibility.json" if root is not None else None
        self._ranges: dict[str, list[VisibleRange]] = {}
        self._load()

    def record(
        self,
        *,
        path: str,
        snapshot_id: str,
        start_line: int,
        end_line: int,
        source: Literal["read_file", "search_text", "edit_file"],
    ) -> None:
        key = self._key(path, snapshot_id)
        values = [
            *self._ranges.get(key, []),
            VisibleRange(
                path=path,
                snapshot_id=snapshot_id,
                start_line=start_line,
                end_line=end_line,
                source=source,
            ),
        ]
        values.sort(key=lambda value: (value.start_line, value.end_line))
        merged: list[VisibleRange] = []
        for value in values:
            if merged and value.start_line <= merged[-1].end_line + 1:
                prior = merged[-1]
                merged[-1] = prior.model_copy(
                    update={"end_line": max(prior.end_line, value.end_line)}
                )
            else:
                merged.append(value)
        self._ranges[key] = merged
        self._save()

    def require(self, *, path: str, snapshot_id: str, start_line: int, end_line: int) -> None:
        ranges = self._ranges.get(self._key(path, snapshot_id), [])
        if not any(
            visible.start_line <= start_line and visible.end_line >= end_line for visible in ranges
        ):
            raise EditError(
                "unseen_range",
                f"Target lines {start_line}-{end_line} were not shown for snapshot "
                f"{snapshot_id[:8].upper()}; read that range first",
            )

    def has_snapshot_view(self, *, path: str, snapshot_id: str) -> bool:
        return bool(self._ranges.get(self._key(path, snapshot_id)))

    @staticmethod
    def _key(path: str, snapshot_id: str) -> str:
        return f"{path}\0{snapshot_id}"

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._ranges = {
                key: [VisibleRange.model_validate(item) for item in values]
                for key, values in payload.items()
            }
        except (OSError, ValueError, json.JSONDecodeError):
            self._ranges = {}

    def _save(self) -> None:
        if self.path is None:
            return
        payload = {
            key: [value.model_dump(mode="json") for value in values]
            for key, values in self._ranges.items()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_bytes(
            self.path,
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
