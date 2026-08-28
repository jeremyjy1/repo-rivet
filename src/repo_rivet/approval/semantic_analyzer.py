"""Derive provider-independent semantic facts from complete tool requests."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from repo_rivet.approval.models import (
    AnalysisLevel,
    ApprovalFacts,
    ApprovalRequest,
    ArtifactProvenance,
    Capability,
    EffectScope,
    ExecutableOrigin,
    OperationClass,
    PathClass,
)

if TYPE_CHECKING:
    from repo_rivet.memory.models import ArtifactRecord, MemoryState

_COMPILERS = frozenset({"c++", "cc", "clang", "clang++", "g++", "gcc", "rustc"})
_GENERATORS = frozenset({"protoc"})
_STATIC_CHECKERS = frozenset({"clang-tidy", "eslint", "mypy", "pyright", "ruff"})
_TEST_RUNNERS = frozenset({"ctest", "pytest"})
_PACKAGE_MANAGERS = frozenset({"npm", "pnpm", "yarn"})
_INSTALL_ACTIONS = frozenset({"add", "install", "sync", "update", "upgrade"})
_DELETE_PROGRAMS = frozenset({"rm", "rmdir", "unlink"})
_NETWORK_PROGRAMS = frozenset({"curl", "ftp", "nc", "rsync", "scp", "ssh", "wget"})
_PRIVILEGED_PROGRAMS = frozenset({"doas", "su", "sudo"})
_OPAQUE_COMPILER_PREFIXES = (
    "@",
    "-fplugin",
    "-specs",
    "--plugin",
    "-wrapper",
    "-b",
    "-dump",
    "-fprofile",
    "-wl,",
    "-xlinker",
)
_OPAQUE_COMPILER_FLAGS = frozenset({"--coverage", "-m", "-md", "-mf", "-mj", "-mmd", "-save-temps"})
_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".m", ".mm", ".rs"})
_CONFIG_NAMES = frozenset(
    {
        "cargo.toml",
        "package.json",
        "pyproject.toml",
        "pytest.ini",
        "setup.cfg",
        "tox.ini",
    }
)
_MANAGED_PARTS = frozenset({".reporivet", ".tmp", "build", "cache", "dist", "target"})
_SAFE_PACKAGE_TEST_PROGRAMS = frozenset({"ava", "jest", "mocha", "pytest", "tap", "vitest"})
_PROTO_OUT_PREFIXES = frozenset(
    {
        "--cpp_out=",
        "--csharp_out=",
        "--java_out=",
        "--js_out=",
        "--kotlin_out=",
        "--objc_out=",
        "--php_out=",
        "--python_out=",
        "--ruby_out=",
    }
)
_IMPORTANT_CAPABILITY_EFFECTS = {
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


class ApprovalFactAnalyzer:
    """Resolve executable identity, operation semantics, paths, and output provenance."""

    def __init__(
        self,
        memory: MemoryState | None = None,
        *,
        trusted_executable_directories: list[str] | None = None,
    ) -> None:
        self.memory = memory
        configured = trusted_executable_directories or []
        defaults = [
            Path(sys.executable).resolve(strict=False).parent,
            Path("/bin"),
            Path("/usr/bin"),
            Path("/usr/local/bin"),
            Path("/opt/homebrew/bin"),
        ]
        self.trusted_executable_directories = {
            Path(path).expanduser().resolve(strict=False) for path in [*defaults, *configured]
        }

    def bind(self, memory: MemoryState) -> None:
        self.memory = memory

    def analyze(self, request: ApprovalRequest) -> ApprovalFacts:
        facts = self._typed_facts(request)
        if request.tool_name in {"run_command", "run_verification"}:
            facts = self._command_facts(request)
        facts.explicit_effects.update(
            _IMPORTANT_CAPABILITY_EFFECTS[capability]
            for capability in request.declared_capabilities
            if capability in _IMPORTANT_CAPABILITY_EFFECTS
        )
        facts.touches_sensitive_paths = any(
            path_class == PathClass.SENSITIVE for path_class in facts.path_classes.values()
        )
        workspace = Path(request.workspace)
        facts.outside_workspace = any(
            not Path(path).is_relative_to(workspace)
            for path in [*facts.read_paths, *facts.write_paths, *facts.delete_paths]
        )
        if facts.touches_sensitive_paths:
            facts.explicit_effects.add("sensitive_file_access")
        if facts.outside_workspace:
            facts.explicit_effects.add("outside_workspace")
        request.facts = facts
        return facts

    def _typed_facts(self, request: ApprovalRequest) -> ApprovalFacts:
        paths = sorted(request.normalized_arguments.get("_resolved_paths", {}).values())
        path_classes = {path: classify_path(Path(path), Path(request.workspace)) for path in paths}
        operation = OperationClass.UNKNOWN
        read_paths: list[str] = []
        write_paths: list[str] = []
        effects: set[str] = set()
        constraints = {"typed_tool"}
        if request.tool_name in {"list_files", "search_text", "read_file", "git_diff"}:
            operation = OperationClass.READ
            read_paths = paths
            effects.add("filesystem_read")
            constraints.add("workspace_path_policy")
        elif request.tool_name in {"write_file", "edit_file"}:
            operation = OperationClass.EDIT
            write_paths = paths
            effects.add("filesystem_write")
            constraints.update({"workspace_path_policy", "snapshot_precondition"})
        scope = effect_scope(write_paths or read_paths, Path(request.workspace))
        return ApprovalFacts(
            operation_class=operation,
            analysis_level=AnalysisLevel.EXACT,
            read_paths=read_paths,
            write_paths=write_paths,
            path_classes=path_classes,
            effect_scope=scope,
            explicit_effects=effects,
            constraints=constraints,
            overwrites_existing=any(Path(path).exists() for path in write_paths),
            reversible=operation == OperationClass.EDIT,
        )

    def _command_facts(self, request: ApprovalRequest) -> ApprovalFacts:
        command = request.normalized_arguments.get("command")
        if not isinstance(command, dict):
            return ApprovalFacts(reasons=["command could not be normalized"])
        raw_program = str(command.get("program", ""))
        program = Path(raw_program).name.lower()
        raw_args = command.get("args", [])
        args = [item for item in raw_args if isinstance(item, str)]
        cwd = _command_cwd(request)
        resolved = resolve_executable(raw_program, cwd)
        origin, artifact = self._executable_origin(raw_program, resolved, request)
        operation, analysis, reasons = self._classify_operation(
            request, program, args, origin, artifact
        )
        if any(not isinstance(item, str) for item in raw_args):
            operation = OperationClass.UNKNOWN
            analysis = AnalysisLevel.OPAQUE
            reasons = ["command contains redacted arguments that cannot be analyzed locally"]
        read_paths, write_paths = _command_paths(operation, program, args, cwd)
        path_classes = {
            path: classify_path(Path(path), Path(request.workspace))
            for path in [*read_paths, *write_paths]
        }
        provenance = {path: self._output_provenance(Path(path), request) for path in write_paths}
        scope = effect_scope(write_paths or read_paths, Path(request.workspace))
        effects, potential = _operation_effects(operation)
        requires_privilege = program in _PRIVILEGED_PROGRAMS
        if requires_privilege:
            effects.add("privilege_escalation")
        constraints = {"captured_output", "shell_free_argv", "stdin_disabled"}
        if cwd.is_relative_to(Path(request.workspace)):
            constraints.add("workspace_cwd")
        timeout = request.normalized_arguments.get("timeout_seconds")
        if isinstance(timeout, (int, float)):
            constraints.add(f"timeout_{timeout:g}")
        if write_paths:
            constraints.add("declared_output_paths")
        verification_kind = None
        if request.tool_name == "run_verification":
            verification_kind = self._verification_kind(request)
            effects.add("registered_verification")
        expanded_command: list[str] = []
        if program in _PACKAGE_MANAGERS:
            script = _package_script(request, program, args)
            if script is not None:
                try:
                    expanded_command = shlex.split(script)
                except ValueError:
                    expanded_command = []
        return ApprovalFacts(
            operation_class=operation,
            analysis_level=analysis,
            executable=raw_program or None,
            resolved_executable=resolved,
            executable_origin=origin,
            expanded_command=expanded_command,
            read_paths=read_paths,
            write_paths=write_paths,
            path_classes=path_classes,
            effect_scope=scope,
            output_provenance=provenance,
            explicit_effects=effects,
            potential_capabilities=potential,
            constraints=constraints,
            accesses_network=operation in {OperationClass.INSTALL, OperationClass.NETWORK},
            requires_privilege=requires_privilege,
            outside_workspace=scope == EffectScope.OUTSIDE_WORKSPACE,
            overwrites_existing=any(Path(path).exists() for path in write_paths),
            reversible=bool(write_paths),
            verification_kind=verification_kind,
            task_relevance="required" if request.tool_name == "run_verification" else "helpful",
            reasons=reasons,
        )

    def _classify_operation(
        self,
        request: ApprovalRequest,
        program: str,
        args: list[str],
        origin: ExecutableOrigin,
        artifact: ArtifactRecord | None,
    ) -> tuple[OperationClass, AnalysisLevel, list[str]]:
        lowered = [item.lower() for item in args]
        if any(item.startswith("@") for item in args):
            return (
                OperationClass.UNKNOWN,
                AnalysisLevel.OPAQUE,
                ["request contains a response file whose arguments are not expanded"],
            )
        if program in _PRIVILEGED_PROGRAMS:
            return (
                OperationClass.UNKNOWN,
                AnalysisLevel.OPAQUE,
                ["command requests privilege escalation"],
            )
        if program in _COMPILERS and any(_is_opaque_compiler_argument(item) for item in args):
            return (
                OperationClass.UNKNOWN,
                AnalysisLevel.OPAQUE,
                ["compiler request contains a plugin, wrapper, or undeclared output option"],
            )
        if program in _DELETE_PROGRAMS:
            return OperationClass.DELETE, AnalysisLevel.EXACT, ["command deletes files"]
        if program in _NETWORK_PROGRAMS:
            return OperationClass.NETWORK, AnalysisLevel.EXACT, ["command accesses the network"]
        if program == "git" and _git_writes(lowered):
            return OperationClass.GIT_WRITE, AnalysisLevel.EXACT, ["command modifies git state"]
        if program in _PACKAGE_MANAGERS:
            if set(lowered) & _INSTALL_ACTIONS:
                return (
                    OperationClass.INSTALL,
                    AnalysisLevel.EXPANDED,
                    ["package action may download and execute third-party code"],
                )
            script = _package_script(request, program, args)
            if script is None:
                return (
                    OperationClass.UNKNOWN,
                    AnalysisLevel.OPAQUE,
                    ["package script could not be resolved"],
                )
            script_program = _first_script_program(script)
            if (
                _package_test_name(program, args)
                and script_program in _SAFE_PACKAGE_TEST_PROGRAMS
                and _package_script_is_single_command(script)
            ):
                return (
                    OperationClass.TEST,
                    AnalysisLevel.EXPANDED,
                    [f"expanded package test script to {script_program}"],
                )
            return (
                OperationClass.UNKNOWN,
                AnalysisLevel.OPAQUE,
                ["expanded package script has unclassified effects"],
            )
        if program in _COMPILERS:
            return (
                OperationClass.BUILD,
                AnalysisLevel.EXACT,
                ["direct compiler invocation with explicit arguments"],
            )
        if program in _GENERATORS:
            output_flags = [item for item in lowered if "_out=" in item]
            if any(item.startswith("--plugin") for item in lowered) or any(
                not item.startswith(tuple(_PROTO_OUT_PREFIXES)) or ":" in item.split("=", 1)[1]
                for item in output_flags
            ):
                return (
                    OperationClass.UNKNOWN,
                    AnalysisLevel.OPAQUE,
                    ["generator loads a custom executable plugin"],
                )
            return (
                OperationClass.GENERATE,
                AnalysisLevel.EXPANDED,
                ["recognized generator with explicit input and output directories"],
            )
        if program in _STATIC_CHECKERS:
            if program == "ruff" and (not args or args[0] not in {"check"}):
                return OperationClass.FORMAT, AnalysisLevel.EXACT, ["formatter may edit sources"]
            if "--fix" in lowered or "--write" in lowered:
                return OperationClass.FORMAT, AnalysisLevel.EXACT, ["checker enables source fixes"]
            return OperationClass.STATIC_CHECK, AnalysisLevel.EXACT, ["static analysis command"]
        if program in _TEST_RUNNERS:
            return OperationClass.TEST, AnalysisLevel.EXACT, ["test runner command"]
        if program in {"python", "python3"} and lowered[:2] == ["-m", "pytest"]:
            return OperationClass.TEST, AnalysisLevel.EXACT, ["python invokes pytest directly"]
        if program == "cargo" and lowered:
            operation = {
                "build": OperationClass.BUILD,
                "check": OperationClass.STATIC_CHECK,
                "test": OperationClass.TEST,
            }.get(lowered[0])
            if operation is not None:
                return operation, AnalysisLevel.EXPANDED, [f"recognized cargo {lowered[0]}"]
        if program == "go" and lowered:
            operation = {
                "build": OperationClass.BUILD,
                "test": OperationClass.TEST,
            }.get(lowered[0])
            if operation is not None:
                return operation, AnalysisLevel.EXPANDED, [f"recognized go {lowered[0]}"]
        if origin == ExecutableOrigin.SESSION_GENERATED and artifact is not None:
            if not _artifact_arguments_are_bounded(args, Path(request.workspace)):
                return (
                    OperationClass.UNKNOWN,
                    AnalysisLevel.OPAQUE,
                    ["session artifact arguments contain paths, URLs, or effect-like options"],
                )
            return (
                OperationClass.RUN_ARTIFACT,
                AnalysisLevel.EXACT,
                ["executable is a current artifact created by this session"],
            )
        return (
            OperationClass.UNKNOWN,
            AnalysisLevel.OPAQUE,
            ["executable semantics are not understood sufficiently for automatic approval"],
        )

    def _executable_origin(
        self,
        requested_program: str,
        resolved: str | None,
        request: ApprovalRequest,
    ) -> tuple[ExecutableOrigin, ArtifactRecord | None]:
        if resolved is None:
            return ExecutableOrigin.UNKNOWN, None
        path = Path(resolved).resolve(strict=False)
        workspace = Path(request.workspace)
        record = self._valid_artifact(path, request)
        if record is not None:
            return ExecutableOrigin.SESSION_GENERATED, record
        is_trusted_location = path.parent in self.trusted_executable_directories
        if path.is_absolute() and path.exists() and is_trusted_location:
            name = Path(requested_program).name.lower()
            if (
                name
                in _COMPILERS
                | _STATIC_CHECKERS
                | _TEST_RUNNERS
                | _PACKAGE_MANAGERS
                | {
                    "cargo",
                    "go",
                    "python",
                    "python3",
                }
                | _GENERATORS
            ):
                return ExecutableOrigin.TRUSTED_TOOLCHAIN, None
        if path.is_relative_to(workspace):
            return ExecutableOrigin.WORKSPACE, None
        if path.is_absolute() and path.exists():
            return ExecutableOrigin.SYSTEM, None
        return ExecutableOrigin.UNKNOWN, None

    def _valid_artifact(
        self,
        path: Path,
        request: ApprovalRequest,
    ) -> ArtifactRecord | None:
        if self.memory is None:
            return None
        key = _artifact_key(path, Path(request.workspace))
        record = self.memory.artifact_registry.get(key)
        if record is None or record.created_by_session != request.session_id:
            return None
        if record.workspace_revision != self.memory.workspace_revision:
            return None
        return record if self._artifact_content_matches(path, record) else None

    @staticmethod
    def _artifact_content_matches(path: Path, record: ArtifactRecord) -> bool:
        if not path.is_file():
            return False
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False
        return digest == record.content_sha256

    def _output_provenance(
        self,
        path: Path,
        request: ApprovalRequest,
    ) -> ArtifactProvenance:
        if not path.exists():
            return ArtifactProvenance.NEW
        if self.memory is None:
            return ArtifactProvenance.USER_FILE
        key = _artifact_key(path, Path(request.workspace))
        record = self.memory.artifact_registry.get(key)
        if record is None or record.created_by_session != request.session_id:
            return ArtifactProvenance.USER_FILE
        if not self._artifact_content_matches(path, record):
            return ArtifactProvenance.USER_FILE
        if record.workspace_revision != self.memory.workspace_revision:
            return ArtifactProvenance.STALE
        return ArtifactProvenance.SESSION_GENERATED

    def _verification_kind(self, request: ApprovalRequest) -> str | None:
        check_id = request.normalized_arguments.get("check_id")
        plan = self.memory.verification_plan if self.memory is not None else None
        if not isinstance(check_id, str) or plan is None:
            return None
        check = next((item for item in plan.checks if item.check_id == check_id), None)
        return check.kind.value if check is not None else None


def resolve_executable(program: str, cwd: Path) -> str | None:
    if not program:
        return None
    candidate = Path(program)
    if candidate.is_absolute():
        return str(candidate.resolve(strict=False))
    if "/" in program:
        return str((cwd / candidate).resolve(strict=False))
    resolved = shutil.which(program)
    return str(Path(resolved).resolve(strict=False)) if resolved is not None else None


def classify_path(path: Path, workspace: Path) -> PathClass:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if name in {".env", ".npmrc", ".pypirc", "credentials"} or name.startswith(".env."):
        return PathClass.SENSITIVE
    if parts & {".aws", ".gnupg", ".ssh"}:
        return PathClass.SENSITIVE
    if name in _CONFIG_NAMES:
        return PathClass.CONFIG
    if "test" in name or "tests" in parts:
        return PathClass.TEST
    if path.suffix.lower() in _SOURCE_SUFFIXES | {".py", ".js", ".ts", ".java"}:
        return PathClass.SOURCE
    if parts & _MANAGED_PARTS:
        if "cache" in parts or ".cache" in parts:
            return PathClass.CACHE
        if ".tmp" in parts or "tmp" in parts:
            return PathClass.TEMP
        return PathClass.BUILD
    if path.is_relative_to(workspace):
        return PathClass.USER_DATA if path.exists() else PathClass.GENERATED
    return PathClass.UNKNOWN


def effect_scope(paths: list[str], workspace: Path) -> EffectScope:
    if not paths:
        return EffectScope.NONE
    candidates = [Path(path) for path in paths]
    if any(not path.is_relative_to(workspace) for path in candidates):
        return EffectScope.OUTSIDE_WORKSPACE
    managed_classes = {PathClass.BUILD, PathClass.CACHE, PathClass.TEMP}
    if all(classify_path(path, workspace) in managed_classes for path in candidates):
        return EffectScope.MANAGED
    return EffectScope.WORKSPACE


def artifact_key(path: Path, workspace: Path) -> str:
    return _artifact_key(path, workspace)


def _artifact_key(path: Path, workspace: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError:
        return str(resolved)


def _command_cwd(request: ApprovalRequest) -> Path:
    value = request.normalized_arguments.get("_resolved_paths", {}).get("cwd", request.workspace)
    return Path(value) if isinstance(value, str) else Path(request.workspace)


def _is_opaque_compiler_argument(argument: str) -> bool:
    lowered = argument.lower()
    return (
        lowered.startswith(_OPAQUE_COMPILER_PREFIXES)
        or lowered in _OPAQUE_COMPILER_FLAGS
        or lowered in {"-xclang", "-load"}
    )


def _command_paths(
    operation: OperationClass,
    program: str,
    args: list[str],
    cwd: Path,
) -> tuple[list[str], list[str]]:
    if operation in {OperationClass.STATIC_CHECK, OperationClass.TEST, OperationClass.FORMAT}:
        ignored_words = {"check", "run", "test", "pytest"}
        inputs = []
        for argument in args:
            if argument.startswith("-") or argument.lower() in ignored_words:
                continue
            candidate = (cwd / argument).resolve(strict=False)
            if (
                argument in {".", ".."}
                or argument.startswith(("./", "../", "/"))
                or candidate.exists()
                or Path(argument).suffix
            ):
                inputs.append(str(candidate))
        if program in _PACKAGE_MANAGERS:
            inputs.append(str((cwd / "package.json").resolve(strict=False)))
        return sorted(set(inputs)), []
    if operation == OperationClass.GENERATE:
        inputs = [
            str((cwd / argument).resolve(strict=False))
            for argument in args
            if not argument.startswith("-") and Path(argument).suffix.lower() == ".proto"
        ]
        outputs = [
            str((cwd / argument.split("=", 1)[1]).resolve(strict=False))
            for argument in args
            if argument.startswith("--") and "_out=" in argument and argument.split("=", 1)[1]
        ]
        return sorted(set(inputs)), sorted(set(outputs))
    if operation != OperationClass.BUILD:
        return [], []
    inputs: list[str] = []
    outputs: list[str] = []
    expect_output = False
    expect_input_path = False
    for argument in args:
        if expect_input_path:
            inputs.append(str((cwd / argument).resolve(strict=False)))
            expect_input_path = False
        elif expect_output:
            outputs.append(str((cwd / argument).resolve(strict=False)))
            expect_output = False
        elif argument == "-o":
            expect_output = True
        elif argument.lower() in {
            "-i",
            "-include",
            "-imacros",
            "-isystem",
            "-isysroot",
            "-l",
        }:
            expect_input_path = True
        elif argument.startswith(("-I", "-L")) and len(argument) > 2:
            inputs.append(str((cwd / argument[2:]).resolve(strict=False)))
        elif argument.startswith("--sysroot="):
            inputs.append(str((cwd / argument.split("=", 1)[1]).resolve(strict=False)))
        elif argument.startswith("-o") and len(argument) > 2:
            outputs.append(str((cwd / argument[2:]).resolve(strict=False)))
        elif not argument.startswith("-") and Path(argument).suffix.lower() in _SOURCE_SUFFIXES | {
            ".a",
            ".dylib",
            ".o",
            ".so",
        }:
            inputs.append(str((cwd / argument).resolve(strict=False)))
    if program == "rustc" and not outputs and inputs:
        outputs.append(str((cwd / Path(inputs[0]).stem).resolve(strict=False)))
    return sorted(set(inputs)), sorted(set(outputs))


def _operation_effects(operation: OperationClass) -> tuple[set[str], set[str]]:
    effects = {"process_execution"}
    potential: set[str] = set()
    if operation == OperationClass.BUILD:
        effects.update({"filesystem_read", "filesystem_write", "compile_workspace_code"})
        potential.add("execute_toolchain_plugins")
    elif operation in {OperationClass.STATIC_CHECK, OperationClass.TEST}:
        effects.update({"filesystem_read", "execute_project_code"})
        potential.update({"filesystem_write", "network_access"})
    elif operation == OperationClass.RUN_ARTIFACT:
        effects.add("execute_session_artifact")
        potential.update({"filesystem_write", "network_access"})
    elif operation == OperationClass.FORMAT:
        effects.update({"filesystem_read", "filesystem_write"})
    elif operation == OperationClass.GENERATE:
        effects.update({"filesystem_read", "filesystem_write", "generate_artifacts"})
    elif operation == OperationClass.INSTALL:
        effects.update(
            {
                "execute_install_scripts",
                "filesystem_write",
                "network_access",
                "package_installation",
            }
        )
    elif operation == OperationClass.DELETE:
        effects.add("filesystem_delete")
    elif operation == OperationClass.GIT_WRITE:
        effects.add("git_write")
    elif operation == OperationClass.NETWORK:
        effects.add("network_access")
    else:
        potential.update({"filesystem_write", "network_access", "dynamic_code_execution"})
    return effects, potential


def _git_writes(args: list[str]) -> bool:
    subcommand = next((item for item in args if not item.startswith("-")), "")
    return subcommand not in {"diff", "log", "show", "status"}


def _package_test_name(program: str, args: list[str]) -> str | None:
    if not args:
        return None
    if program in {"npm", "pnpm"} and args[0] == "run" and len(args) > 1:
        return args[1] if args[1] == "test" else None
    return args[0] if args[0] == "test" else None


def _package_script(request: ApprovalRequest, program: str, args: list[str]) -> str | None:
    script_name = _package_test_name(program, args)
    if script_name is None:
        return None
    try:
        payload = json.loads((_command_cwd(request) / "package.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    scripts = payload.get("scripts") if isinstance(payload, dict) else None
    script = scripts.get(script_name) if isinstance(scripts, dict) else None
    return script if isinstance(script, str) else None


def _first_script_program(script: str) -> str:
    first_stage = script.split("&&", 1)[0].split("||", 1)[0].split(";", 1)[0].strip()
    return Path(first_stage.split(maxsplit=1)[0]).name.lower() if first_stage else ""


def _package_script_is_single_command(script: str) -> bool:
    if any(operator in script for operator in ("&&", "||", ";", "|", ">", "<", "$(", "`")):
        return False
    lowered = script.lower()
    return not any(flag in lowered for flag in ("--coverage", "--outputfile", "--update"))


def _artifact_arguments_are_bounded(arguments: list[str], workspace: Path) -> bool:
    for argument in arguments:
        lowered = argument.lower()
        if "://" in lowered or lowered in {"--delete", "--install", "--network", "--write"}:
            return False
        candidate = Path(argument)
        if candidate.is_absolute() or ".." in candidate.parts:
            return False
        if candidate.suffix and (workspace / candidate).exists():
            return False
    return True
