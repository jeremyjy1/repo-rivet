"""Build an isolated, structured input for the independent approval reviewer."""

from __future__ import annotations

from typing import Any

from repo_rivet.approval.models import ApprovalRequest

IMPORTANT_EFFECTS = frozenset(
    {
        "filesystem_write",
        "filesystem_delete",
        "network_access",
        "sensitive_file_access",
        "privilege_escalation",
        "outside_workspace",
        "dynamic_code_execution",
        "package_installation",
        "git_write",
        "git_history_rewrite",
        "interactive_shell",
        "execute_remote_content",
    }
)

AUTO_APPROVAL_BLOCKING_EFFECTS = IMPORTANT_EFFECTS - {"filesystem_write"}


def build_review_payload(request: ApprovalRequest) -> dict[str, Any]:
    """Create reviewer input from local facts without conversation history or secrets."""
    facts = request.facts
    return {
        "task": {
            "summary": request.task_summary or "Task summary unavailable",
            "requested_by_user": bool(request.task_summary),
            "local_relevance": facts.task_relevance,
        },
        "workspace": {
            "root": request.workspace,
            "trust_level": "user_selected_repository",
        },
        "operation": {
            "class": facts.operation_class.value,
            "analysis_level": facts.analysis_level.value,
            "executable": facts.executable,
            "resolved_executable": facts.resolved_executable,
            "executable_origin": facts.executable_origin.value,
            "expanded_command": facts.expanded_command,
            "verification_kind": facts.verification_kind,
            "semantic_reasons": facts.reasons,
        },
        "execution_plan": _execution_plan(request),
        "deterministic_effects": {
            "capabilities": sorted(facts.explicit_effects),
            "potential_capabilities": sorted(facts.potential_capabilities),
            "read_paths": facts.read_paths,
            "write_paths": facts.write_paths,
            "delete_paths": facts.delete_paths,
            "path_classes": {
                path: path_class.value for path, path_class in facts.path_classes.items()
            },
            "effect_scope": facts.effect_scope.value,
            "output_provenance": {
                path: provenance.value for path, provenance in facts.output_provenance.items()
            },
            "network_access": facts.accesses_network,
            "privilege_escalation": facts.requires_privilege,
            "outside_workspace": facts.outside_workspace,
            "sensitive_file_access": facts.touches_sensitive_paths,
            "overwrites_existing_files": facts.overwrites_existing,
            "reversible": facts.reversible,
        },
        "available_constraints": sorted(facts.constraints),
    }


def _execution_plan(request: ApprovalRequest) -> dict[str, Any]:
    if request.tool_name in {"write_file", "edit_file", "delete_path"}:
        return _file_change_plan(request)
    command = request.normalized_arguments.get("command")
    if not isinstance(command, dict):
        return {"tool": request.tool_name}
    stage: dict[str, Any] = {
        "program": command.get("program"),
        "resolved_program": request.facts.resolved_executable,
        "args": command.get("args", []),
        "cwd": request.normalized_arguments.get("_resolved_paths", {}).get(
            "cwd", request.workspace
        ),
    }
    if request.facts.analysis_level.value == "expanded":
        stage["semantic_context"] = {
            "analysis_level": "expanded",
            "expanded_command": request.facts.expanded_command,
            "reason": request.facts.reasons,
        }
    return {
        "stages": [stage],
        "stdin": {
            "type": "provided" if request.normalized_arguments.get("stdin") is not None else "null"
        },
        "stdout": {"type": "capture"},
        "stderr": {"type": "capture"},
        "timeout_seconds": request.normalized_arguments.get("timeout_seconds"),
    }


def _file_change_plan(request: ApprovalRequest) -> dict[str, Any]:
    normalized = request.normalized_arguments
    resolved_paths = normalized.get("_resolved_paths", {})
    target = resolved_paths.get("path") if isinstance(resolved_paths, dict) else None
    plan: dict[str, Any] = {
        "tool": request.tool_name,
        "operation": (
            "create"
            if request.tool_name == "write_file"
            else "delete"
            if request.tool_name == "delete_path"
            else "edit"
        ),
        "target": target,
        "overwrites_existing_file": request.facts.overwrites_existing,
        "constraints": sorted(request.facts.constraints),
    }
    if request.tool_name == "write_file":
        content = request.arguments.get("content")
        if isinstance(content, str):
            plan["content_summary"] = {
                "characters": len(content),
                "lines": len(content.splitlines()),
            }
        return plan

    if request.tool_name == "delete_path":
        plan["recursive"] = bool(normalized.get("recursive"))
        plan["entry_type"] = normalized.get("entry_type")
        plan["entry_count"] = normalized.get("entry_count")
        plan["total_bytes"] = normalized.get("total_bytes")
        return plan

    operations = normalized.get("operations")
    if isinstance(operations, list):
        plan["operations"] = [
            {
                key: value
                for key, value in operation.items()
                if key
                in {
                    "op",
                    "start_line",
                    "end_line",
                    "line",
                    "new_line_count",
                }
            }
            for operation in operations
            if isinstance(operation, dict)
        ]
    diff_preview = normalized.get("diff_preview")
    if isinstance(diff_preview, str):
        plan["diff_preview"] = _bounded_text(diff_preview)
    return plan


def _bounded_text(value: str, limit: int = 12_000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n... preview truncated by approval context limit ..."
