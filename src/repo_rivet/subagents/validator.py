"""Local validation for evidence-backed child reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from repo_rivet.editing.snapshot_store import SnapshotStore
from repo_rivet.memory.models import MemoryState
from repo_rivet.subagents.models import (
    DelegationRequest,
    SubagentReport,
    ValidatedSubagentReport,
)
from repo_rivet.subagents.policy import ScopedWorkspacePathPolicy


class SubagentResultValidator:
    def __init__(
        self,
        workspace: Path,
        *,
        max_report_chars: int = 4_000,
        max_files_read: int = 12,
    ) -> None:
        self.workspace = workspace.resolve()
        self.max_report_chars = max_report_chars
        self.max_files_read = max_files_read

    def validate(
        self,
        report: SubagentReport,
        *,
        request: DelegationRequest,
        child_memory: MemoryState,
        child_directory: Path,
        parent_memory: MemoryState,
    ) -> ValidatedSubagentReport:
        if report.delegation_id != request.delegation_id:
            raise ValueError("Subagent report does not match its delegation")
        if report.base_workspace_revision != request.base_workspace_revision:
            raise ValueError("Subagent report uses the wrong base workspace revision")
        serialized = json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(serialized) > self.max_report_chars:
            raise ValueError(f"Subagent report exceeds {self.max_report_chars} characters")
        if child_memory.modified_files or (
            child_memory.workspace_revision != request.base_workspace_revision
        ):
            raise ValueError("Read-only subagent changed workspace state")

        policy = ScopedWorkspacePathPolicy(
            self.workspace,
            allowed_paths=request.scope_paths,
            excluded_paths=request.excluded_paths,
        )
        known_evidence = {
            *(event.event_id for event in child_memory.observation_events),
            *request.evidence_refs,
        }
        for finding in report.findings:
            missing = set(finding.evidence_refs) - known_evidence
            if missing:
                raise ValueError(
                    "Subagent finding references unknown evidence: " + ", ".join(sorted(missing))
                )
            for path in finding.affected_paths:
                policy.resolve(path)

        verification_refs = set(report.verification_refs)
        allowed_verification_refs = set(request.evidence_refs) & set(
            parent_memory.verification_results
        )
        if not verification_refs.issubset(allowed_verification_refs):
            missing = verification_refs - allowed_verification_refs
            raise ValueError(
                "Subagent report references undelegated verification results: "
                + ", ".join(sorted(missing))
            )

        actual_files = list(dict.fromkeys(child_memory.summary.files_read))
        if len(actual_files) > self.max_files_read:
            raise ValueError(f"Subagent read more than {self.max_files_read} files")
        for path in actual_files:
            policy.resolve(path)

        snapshots = {
            path: snapshot_id
            for path, snapshot_id in child_memory.current_snapshots.items()
            if path in actual_files
        }
        stale_paths = self._stale_snapshot_paths(snapshots, child_directory)
        normalized = report.model_copy(
            update={
                "files_read": actual_files,
                "snapshots_used": snapshots,
            }
        )
        return ValidatedSubagentReport(
            report=normalized,
            freshness="stale" if stale_paths else "fresh",
            stale_paths=stale_paths,
        )

    def _stale_snapshot_paths(
        self,
        snapshots: dict[str, str],
        child_directory: Path,
    ) -> list[str]:
        store = SnapshotStore(child_directory / "snapshots")
        stale: list[str] = []
        for path, snapshot_id in snapshots.items():
            try:
                snapshot = store.get(snapshot_id)
                current_hash = hashlib.sha256((self.workspace / path).read_bytes()).hexdigest()
            except (OSError, ValueError):
                stale.append(path)
                continue
            if snapshot.relative_path != path or snapshot.raw_bytes_hash != current_hash:
                stale.append(path)
        return stale
