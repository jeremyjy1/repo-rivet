"""Tool-specific, revision-aware semantic action identity."""

from __future__ import annotations

import hashlib
import json
import shlex
from typing import Any, Protocol

from repo_rivet.actions.models import ActionRecord, ActionStatus
from repo_rivet.agent.phases import RevisionVector
from repo_rivet.tools.base import ToolCall, ToolResult


class IdentityContext(Protocol):
    current_snapshots: dict[str, str]
    workspace_revision: int


def _digest(value: dict[str, Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _path(arguments: dict[str, Any]) -> str:
    return str(arguments.get("path", "."))


class ActionIdentity:
    """Build stable keys from tool semantics instead of raw call equality."""

    @staticmethod
    def build(
        call: ToolCall,
        *,
        context: IdentityContext,
        revisions: RevisionVector,
        plan_step_id: str | None,
    ) -> str:
        arguments = call.arguments
        tool = call.name
        identity: dict[str, Any] = {"tool": tool}
        if tool == "read_file":
            path = _path(arguments)
            identity.update(
                {
                    "path": path,
                    "snapshot": context.current_snapshots.get(
                        path,
                        f"workspace-{revisions.workspace}",
                    ),
                    "start_line": arguments.get("start_line", 1),
                    "end_line": arguments.get("end_line"),
                }
            )
        elif tool == "edit_file":
            identity.update(
                {
                    "path": _path(arguments),
                    "base_snapshot_id": arguments.get("snapshot_id"),
                    "operations": arguments.get("operations", []),
                }
            )
        elif tool == "write_file":
            path = _path(arguments)
            identity.update(
                {
                    "path": path,
                    "content": arguments.get("content", ""),
                    "base": context.current_snapshots.get(
                        path,
                        f"missing-at-{revisions.workspace}",
                    ),
                }
            )
        elif tool == "delete_path":
            path = _path(arguments)
            identity.update(
                {
                    "path": path,
                    "recursive": bool(arguments.get("recursive", False)),
                    "base": context.current_snapshots.get(
                        path,
                        f"workspace-{revisions.workspace}",
                    ),
                }
            )
        elif tool == "run_verification":
            identity.update(
                {
                    "check_id": arguments.get("check_id"),
                    "workspace_revision": revisions.workspace,
                    "verification_plan_revision": revisions.verification_plan,
                    "environment_revision": revisions.environment,
                }
            )
        elif tool == "run_command":
            command = arguments.get("command")
            if isinstance(command, str):
                try:
                    normalized_command: object = shlex.split(command, posix=True)
                except ValueError:
                    normalized_command = command
            else:
                normalized_command = command
            identity.update(
                {
                    "command": normalized_command,
                    "cwd": arguments.get("cwd", "."),
                    "stdin": arguments.get("stdin"),
                    "workspace_revision": revisions.workspace,
                    "environment_revision": revisions.environment,
                }
            )
        elif tool in {
            "search_text",
            "semantic_query",
            "list_files",
            "git_status",
            "git_diff",
        }:
            identity.update(
                {
                    "arguments": arguments,
                    "workspace_revision": revisions.workspace,
                }
            )
        else:
            identity.update(
                {
                    "arguments": arguments,
                    "workspace_revision": revisions.workspace,
                    "plan_step_id": plan_step_id,
                }
            )
        return f"{tool}:{_digest(identity)}"

    @staticmethod
    def result_key(
        action: ActionRecord,
        result: ToolResult,
        *,
        context: IdentityContext,
        revisions: RevisionVector,
    ) -> str:
        metadata = result.metadata or {}
        if action.tool_name in {"read_file", "write_file"} and result.ok:
            snapshot = metadata.get("snapshot_id") or metadata.get("new_snapshot_id")
            path = str(metadata.get("path") or action.normalized_arguments.get("path", "."))
            if isinstance(snapshot, str):
                arguments = dict(action.normalized_arguments)
                if action.tool_name == "read_file":
                    identity = {
                        "tool": "read_file",
                        "path": path,
                        "snapshot": snapshot,
                        "start_line": arguments.get("start_line", 1),
                        "end_line": arguments.get("end_line"),
                    }
                else:
                    identity = {
                        "tool": "write_file",
                        "path": path,
                        "content": arguments.get("content", ""),
                        "base": snapshot,
                    }
                return f"{action.tool_name}:{_digest(identity)}"
        return action.semantic_key

    @staticmethod
    def result_still_valid(
        action: ActionRecord,
        *,
        context: IdentityContext,
        revisions: RevisionVector,
    ) -> bool:
        if action.result is None:
            return False
        if action.tool_name == "read_file":
            metadata = action.result.metadata or {}
            path = str(metadata.get("path") or action.normalized_arguments.get("path", "."))
            snapshot = metadata.get("snapshot_id")
            return isinstance(snapshot, str) and context.current_snapshots.get(path) == snapshot
        if action.tool_name == "run_verification":
            return (
                action.revisions.workspace == revisions.workspace
                and action.revisions.verification_plan == revisions.verification_plan
                and action.revisions.environment == revisions.environment
            )
        if action.tool_name == "run_command":
            return (
                action.revisions.workspace == revisions.workspace
                and action.revisions.environment == revisions.environment
            )
        if action.tool_name in {"edit_file", "write_file", "delete_path"}:
            # A successfully applied mutation is never replayed. Reusing its observation is
            # the safe interpretation of the same semantic proposal.
            return True
        return action.revisions.workspace == revisions.workspace

    @staticmethod
    def applied_edit_covers(
        call: ToolCall,
        action: ActionRecord,
        *,
        context: IdentityContext,
    ) -> bool:
        """Return whether a stale edit proposal is already present in the live snapshot."""
        if call.name != "edit_file" or action.tool_name != "edit_file":
            return False
        if action.result is None or action.status != ActionStatus.SUCCEEDED:
            return False

        requested = call.arguments
        applied = action.normalized_arguments
        if _path(requested) != _path(applied):
            return False
        if requested.get("snapshot_id") != applied.get("snapshot_id"):
            return False

        metadata = action.result.metadata or {}
        new_snapshot = metadata.get("new_snapshot_id")
        path = _path(requested)
        if not isinstance(new_snapshot, str) or context.current_snapshots.get(path) != new_snapshot:
            return False

        requested_operations = requested.get("operations")
        applied_operations = applied.get("operations")
        if not isinstance(requested_operations, list) or not isinstance(applied_operations, list):
            return False
        return bool(requested_operations) and all(
            isinstance(requested_operation, dict)
            and any(
                isinstance(applied_operation, dict)
                and _edit_operation_covers(applied_operation, requested_operation)
                for applied_operation in applied_operations
            )
            for requested_operation in requested_operations
        )


def _edit_operation_covers(applied: dict[str, Any], requested: dict[str, Any]) -> bool:
    """Compare operations against their shared base snapshot without replaying the edit."""
    if applied == requested:
        return True
    if applied.get("op") != requested.get("op"):
        return False
    operation = applied.get("op")
    if operation not in {"replace", "delete"}:
        return False

    applied_start = applied.get("start_line")
    applied_end = applied.get("end_line")
    requested_start = requested.get("start_line")
    requested_end = requested.get("end_line")
    if not all(
        isinstance(value, int)
        for value in (applied_start, applied_end, requested_start, requested_end)
    ):
        return False
    assert isinstance(applied_start, int)
    assert isinstance(applied_end, int)
    assert isinstance(requested_start, int)
    assert isinstance(requested_end, int)
    if not (applied_start <= requested_start <= requested_end <= applied_end):
        return False
    if operation == "delete":
        return True

    applied_lines = applied.get("new_lines")
    requested_lines = requested.get("new_lines")
    if not isinstance(applied_lines, list) or not isinstance(requested_lines, list):
        return False
    # A positional subset is comparable only when the applied replacement preserved the
    # base range's line count. Insertions and contractions need a fresh snapshot instead.
    if len(applied_lines) != applied_end - applied_start + 1:
        return False
    if len(requested_lines) != requested_end - requested_start + 1:
        return False
    offset = requested_start - applied_start
    return requested_lines == applied_lines[offset : offset + len(requested_lines)]
