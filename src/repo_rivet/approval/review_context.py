"""Build local, structured facts for the independent approval reviewer."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from repo_rivet.approval.models import ApprovalRequest, Capability

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

_CAPABILITY_EFFECTS = {
    Capability.FILESYSTEM_READ: "filesystem_read",
    Capability.FILESYSTEM_WRITE: "filesystem_write",
    Capability.FILESYSTEM_DELETE: "filesystem_delete",
    Capability.PROCESS_EXECUTE: "process_execution",
    Capability.NETWORK_ACCESS: "network_access",
    Capability.SECRET_READ: "sensitive_file_access",
    Capability.OUTSIDE_WORKSPACE: "outside_workspace",
    Capability.DEVICE_ACCESS: "device_access",
    Capability.GIT_WRITE: "git_write",
    Capability.GIT_HISTORY_REWRITE: "git_history_rewrite",
    Capability.PRIVILEGE_ESCALATION: "privilege_escalation",
}
_COMPILERS = frozenset({"c++", "cc", "clang", "clang++", "g++", "gcc", "rustc"})
_PROJECT_RUNNERS = frozenset(
    {
        "cargo",
        "ctest",
        "go",
        "make",
        "node",
        "npm",
        "npx",
        "pnpm",
        "pytest",
        "python",
        "python3",
        "tox",
        "uv",
        "yarn",
    }
)
_PACKAGE_MANAGERS = frozenset({"cargo", "npm", "pip", "pip3", "pnpm", "uv", "yarn"})
_PACKAGE_ACTIONS = frozenset({"add", "install", "sync", "update", "upgrade"})
_DYNAMIC_FLAGS = {
    "node": frozenset({"-e", "--eval"}),
    "python": frozenset({"-c"}),
    "python3": frozenset({"-c"}),
}
_INTERACTIVE_PROGRAMS = frozenset({"bash", "fish", "irb", "node", "python", "python3", "sh", "zsh"})


def attach_review_facts(request: ApprovalRequest) -> None:
    """Attach the minimum effects and enforceable constraints found locally."""
    effects = {
        _CAPABILITY_EFFECTS[capability]
        for capability in request.assessment.capabilities
        if capability in _CAPABILITY_EFFECTS
    }
    constraints: set[str] = set()

    if request.tool_name in {"run_command", "run_verification"}:
        effects.update(command_effects(request))
        read_paths, write_paths = command_effect_paths(request)
        workspace = Path(request.workspace)
        if any(not Path(path).is_relative_to(workspace) for path in [*read_paths, *write_paths]):
            effects.add("outside_workspace")
        constraints.update(_command_constraints(request))
    else:
        constraints.add("typed_tool")
        if request.assessment.affected_paths:
            constraints.add("workspace_path_policy")
        if request.tool_name == "write_file":
            constraints.add("create_only")
        elif request.tool_name == "edit_file":
            constraints.update({"atomic_file_replace", "snapshot_precondition"})

    request.deterministic_effects = effects
    request.available_constraints = constraints


def build_review_payload(request: ApprovalRequest) -> dict[str, Any]:
    """Create the isolated reviewer input without conversation history or secrets."""
    capabilities = request.deterministic_effects
    read_paths, write_paths = _effect_paths(request)
    return {
        "task": {
            "summary": request.task_summary or "Task summary unavailable",
            "requested_by_user": bool(request.task_summary),
        },
        "workspace": {
            "root": request.workspace,
            "trust_level": "user_selected_repository",
        },
        "execution_plan": _execution_plan(request),
        "deterministic_effects": {
            "capabilities": sorted(capabilities),
            "read_paths": read_paths,
            "write_paths": write_paths,
            "network_access": "network_access" in capabilities,
            "privilege_escalation": "privilege_escalation" in capabilities,
            "outside_workspace": "outside_workspace" in capabilities,
            "sensitive_file_access": "sensitive_file_access" in capabilities,
            "dynamic_code_execution": "dynamic_code_execution" in capabilities,
            "overwrites_existing_files": request.tool_name == "edit_file"
            or any(Path(path).exists() for path in write_paths),
        },
        "available_constraints": sorted(request.available_constraints),
    }


def command_effects(request: ApprovalRequest) -> set[str]:
    command = request.normalized_arguments.get("command")
    if not isinstance(command, dict):
        return set()
    program = Path(str(command.get("program", ""))).name.lower()
    args = [item for item in command.get("args", []) if isinstance(item, str)]
    lowered = {item.lower() for item in args}
    effects: set[str] = set()

    if program in _COMPILERS:
        effects.update({"filesystem_read", "filesystem_write", "compile_workspace_code"})
    if program in _PROJECT_RUNNERS or _program_is_in_workspace(request, program):
        effects.add("execute_project_code")
    if program in _PACKAGE_MANAGERS and lowered & _PACKAGE_ACTIONS:
        effects.update(
            {
                "filesystem_write",
                "network_access",
                "package_installation",
                "execute_install_scripts",
            }
        )
    if lowered & _DYNAMIC_FLAGS.get(program, frozenset()):
        effects.add("dynamic_code_execution")
    if program in _INTERACTIVE_PROGRAMS and not args:
        effects.add("interactive_shell")
    return effects


def _command_constraints(request: ApprovalRequest) -> set[str]:
    constraints = {"captured_output", "shell_free_argv"}
    cwd = request.normalized_arguments.get("_resolved_paths", {}).get("cwd")
    if isinstance(cwd, str) and Path(cwd).is_relative_to(Path(request.workspace)):
        constraints.add("workspace_cwd")
    timeout = request.normalized_arguments.get("timeout_seconds")
    if isinstance(timeout, (int, float)):
        constraints.add(f"timeout_{timeout:g}")
    stdin = request.normalized_arguments.get("stdin")
    constraints.add("stdin_fixed" if stdin is not None else "stdin_disabled")
    return constraints


def _execution_plan(request: ApprovalRequest) -> dict[str, Any]:
    command = request.normalized_arguments.get("command")
    if not isinstance(command, dict):
        return {
            "tool": request.tool_name,
            "arguments": _public_arguments(request.normalized_arguments),
        }

    program = str(command.get("program", ""))
    args = command.get("args", [])
    stdin = request.normalized_arguments.get("stdin")
    stdin_plan: dict[str, Any] = {"type": "null"}
    if isinstance(stdin, dict):
        stdin_plan = {"type": "provided", "characters": stdin.get("characters", 0)}
    elif stdin is not None:
        stdin_plan = {"type": "provided"}
    stage: dict[str, Any] = {
        "program": program,
        "resolved_program": _resolve_program(request, program),
        "args": args,
        "cwd": request.normalized_arguments.get("_resolved_paths", {}).get(
            "cwd", request.workspace
        ),
    }
    semantic_context = _semantic_context(request, program, args)
    if semantic_context:
        stage["semantic_context"] = semantic_context
    return {
        "stages": [stage],
        "stdin": stdin_plan,
        "stdout": {"type": "capture"},
        "stderr": {"type": "capture"},
        "timeout_seconds": request.normalized_arguments.get("timeout_seconds"),
    }


def _semantic_context(
    request: ApprovalRequest,
    program: str,
    arguments: object,
) -> dict[str, Any]:
    args = (
        [item for item in arguments if isinstance(item, str)] if isinstance(arguments, list) else []
    )
    name = Path(program).name.lower()
    context: dict[str, Any] = {}
    cwd_value = request.normalized_arguments.get("_resolved_paths", {}).get(
        "cwd", request.workspace
    )
    cwd = Path(cwd_value) if isinstance(cwd_value, str) else Path(request.workspace)

    if name in {"npm", "pnpm", "yarn"}:
        script_name = _package_script_name(name, args)
        resolved_script = _read_package_script(cwd, script_name)
        if script_name is not None:
            context["script_name"] = script_name
            context["resolved_script"] = resolved_script
    if name in {"python", "python3", "node"} and args and not args[0].startswith("-"):
        script = (cwd / args[0]).resolve(strict=False)
        context["script_path"] = str(script)
        context["script_in_workspace"] = script.is_relative_to(Path(request.workspace))
    return context


def _package_script_name(program: str, args: list[str]) -> str | None:
    if not args:
        return None
    if program in {"npm", "pnpm"} and args[0] == "run" and len(args) > 1:
        return args[1]
    if args[0] in {"test", "start"}:
        return args[0]
    return None


def _read_package_script(cwd: Path, script_name: str | None) -> str | None:
    if script_name is None:
        return None
    try:
        payload = json.loads((cwd / "package.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    scripts = payload.get("scripts") if isinstance(payload, dict) else None
    script = scripts.get(script_name) if isinstance(scripts, dict) else None
    return script if isinstance(script, str) else None


def _resolve_program(request: ApprovalRequest, program: str) -> str | None:
    if not program:
        return None
    candidate = Path(program)
    if candidate.is_absolute():
        return str(candidate.resolve(strict=False))
    if "/" in program:
        cwd_value = request.normalized_arguments.get("_resolved_paths", {}).get(
            "cwd", request.workspace
        )
        cwd = Path(cwd_value) if isinstance(cwd_value, str) else Path(request.workspace)
        return str((cwd / candidate).resolve(strict=False))
    return shutil.which(program)


def _program_is_in_workspace(request: ApprovalRequest, program: str) -> bool:
    resolved = _resolve_program(request, program)
    return bool(resolved and Path(resolved).is_relative_to(Path(request.workspace)))


def _effect_paths(request: ApprovalRequest) -> tuple[list[str], list[str]]:
    if request.tool_name in {"run_command", "run_verification"}:
        return command_effect_paths(request)
    paths = sorted(request.assessment.affected_paths)
    read_paths = paths if "filesystem_read" in request.deterministic_effects else []
    write_paths = paths if "filesystem_write" in request.deterministic_effects else []
    return read_paths, write_paths


def command_effect_paths(request: ApprovalRequest) -> tuple[list[str], list[str]]:
    command = request.normalized_arguments.get("command")
    if not isinstance(command, dict):
        return [], []
    program = Path(str(command.get("program", ""))).name.lower()
    args = [item for item in command.get("args", []) if isinstance(item, str)]
    if program not in _COMPILERS:
        return [], []
    cwd_value = request.normalized_arguments.get("_resolved_paths", {}).get(
        "cwd", request.workspace
    )
    cwd = Path(cwd_value) if isinstance(cwd_value, str) else Path(request.workspace)
    inputs: list[str] = []
    outputs: list[str] = []
    output_next = False
    for argument in args:
        if output_next:
            outputs.append(str((cwd / argument).resolve(strict=False)))
            output_next = False
        elif argument == "-o":
            output_next = True
        elif not argument.startswith("-") and Path(argument).suffix:
            inputs.append(str((cwd / argument).resolve(strict=False)))
    return sorted(inputs), sorted(outputs)


def _public_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in arguments.items()
        if not key.startswith("_")
        and key not in {"prepared_live_hash", "snapshot_id", "snapshot_tag"}
    }
