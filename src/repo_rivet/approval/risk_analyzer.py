"""Deterministic capability extraction and conservative risk classification."""

from pathlib import Path

from repo_rivet.approval.models import ApprovalRequest, Capability, RiskAssessment, RiskLevel
from repo_rivet.approval.safe_rules import is_obviously_safe

_SENSITIVE_NAMES = frozenset(
    {
        ".env",
        ".npmrc",
        ".pypirc",
        "credentials",
        "id_dsa",
        "id_ed25519",
        "id_rsa",
        "reporivet.toml",
    }
)
_NETWORK_PROGRAMS = frozenset({"curl", "ftp", "nc", "rsync", "scp", "ssh", "wget"})
_DELETE_PROGRAMS = frozenset({"rm", "rmdir", "unlink"})
_PRIVILEGED_PROGRAMS = frozenset({"doas", "su", "sudo"})
_PACKAGE_MANAGERS = frozenset({"cargo", "npm", "pip", "pip3", "pnpm", "uv", "yarn"})
_READ_ONLY_PROGRAMS = frozenset({"pwd", "whoami"})
_SHELL_OPERATORS = frozenset({"&", "&&", ";", "<", "<<", ">", ">>", "|", "||"})


class RiskAnalyzer:
    """Combine declared tool capabilities with request-specific facts."""

    def assess(self, request: ApprovalRequest) -> RiskAssessment:
        capabilities = set(request.declared_capabilities)
        reasons: list[str] = []
        affected_paths = list(
            request.normalized_arguments.get("_resolved_paths", {}).values()
        )
        outside_paths = request.normalized_arguments.get("_outside_workspace_paths", [])
        if outside_paths:
            capabilities.add(Capability.OUTSIDE_WORKSPACE)
            reasons.append("requested path resolves outside the configured workspace")

        sensitive_paths = [path for path in affected_paths if _is_sensitive_path(path)]
        if sensitive_paths and Capability.FILESYSTEM_READ in capabilities:
            capabilities.add(Capability.SECRET_READ)
            reasons.append("request targets a credential or sensitive configuration file")
        if any(_is_device_path(path) for path in affected_paths):
            capabilities.add(Capability.DEVICE_ACCESS)
            reasons.append("request targets a device path")

        level = self._base_level(capabilities)
        if request.tool_name == "run_command":
            level = max(level, self._assess_command(request, capabilities, reasons))

        if (
            Capability.PRIVILEGE_ESCALATION in capabilities
            or Capability.DEVICE_ACCESS in capabilities
        ):
            level = RiskLevel.CRITICAL
        elif (
            Capability.OUTSIDE_WORKSPACE in capabilities
            or Capability.SECRET_READ in capabilities
        ):
            level = max(level, RiskLevel.HIGH)

        if not reasons:
            reasons.extend(self._default_reasons(capabilities))
        assessment = RiskAssessment(
            level=level,
            capabilities=capabilities,
            reasons=reasons,
            affected_paths=affected_paths,
            sensitive_paths=sensitive_paths,
        )
        request.assessment = assessment
        assessment.obviously_safe = is_obviously_safe(request)
        return assessment

    @staticmethod
    def _base_level(capabilities: set[Capability]) -> RiskLevel:
        if Capability.FILESYSTEM_DELETE in capabilities:
            return RiskLevel.HIGH
        if Capability.FILESYSTEM_WRITE in capabilities:
            return RiskLevel.MEDIUM
        if Capability.PROCESS_EXECUTE in capabilities:
            return RiskLevel.MEDIUM
        if Capability.FILESYSTEM_READ in capabilities:
            return RiskLevel.SAFE
        return RiskLevel.LOW

    @staticmethod
    def _assess_command(
        request: ApprovalRequest,
        capabilities: set[Capability],
        reasons: list[str],
    ) -> RiskLevel:
        command = request.normalized_arguments.get("command", {})
        if not isinstance(command, dict):
            reasons.append("command could not be normalized")
            return RiskLevel.HIGH
        program = Path(str(command.get("program", ""))).name.lower()
        args = [str(item) for item in command.get("args", [])]
        lowered = [item.lower() for item in args]
        if any(item in _SHELL_OPERATORS for item in args):
            reasons.append("shell operators are not supported")
            return RiskLevel.HIGH
        if program in _PRIVILEGED_PROGRAMS:
            capabilities.add(Capability.PRIVILEGE_ESCALATION)
            reasons.append("command requests privilege escalation")
            return RiskLevel.CRITICAL
        if program in _NETWORK_PROGRAMS:
            capabilities.add(Capability.NETWORK_ACCESS)
            reasons.append("command can access the network")
            return RiskLevel.HIGH
        if program in _DELETE_PROGRAMS:
            capabilities.add(Capability.FILESYSTEM_DELETE)
            reasons.append("command can delete files")
            return RiskLevel.HIGH
        if program == "git":
            return _assess_git(lowered, capabilities, reasons)
        if program in _PACKAGE_MANAGERS:
            reasons.append("package tooling may download and execute third-party code")
            if any(item in {"add", "install", "sync"} for item in lowered):
                capabilities.add(Capability.NETWORK_ACCESS)
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM
        if program in _READ_ONLY_PROGRAMS and not args:
            reasons.append("command is a narrow read-only process")
            return RiskLevel.SAFE
        reasons.append("command executes workspace or external code")
        return RiskLevel.MEDIUM

    @staticmethod
    def _default_reasons(capabilities: set[Capability]) -> list[str]:
        if Capability.FILESYSTEM_WRITE in capabilities:
            return ["request modifies workspace files"]
        if Capability.FILESYSTEM_READ in capabilities:
            return ["request performs a confined typed read"]
        return ["request uses an unclassified tool capability"]


def _assess_git(
    args: list[str],
    capabilities: set[Capability],
    reasons: list[str],
) -> RiskLevel:
    read_only = {"diff", "log", "show", "status"}
    subcommand = next((item for item in args if not item.startswith("-")), "")
    if subcommand in read_only:
        reasons.append("git command reads repository state")
        return RiskLevel.MEDIUM
    capabilities.add(Capability.GIT_WRITE)
    if subcommand in {"clean", "rebase", "reset"}:
        capabilities.add(Capability.GIT_HISTORY_REWRITE)
        reasons.append("git command may discard or rewrite repository state")
        return RiskLevel.HIGH
    reasons.append("git command modifies repository state")
    return RiskLevel.MEDIUM


def _is_sensitive_path(path: str) -> bool:
    candidate = Path(path)
    name = candidate.name.lower()
    if name in _SENSITIVE_NAMES or name.startswith(".env."):
        return True
    return any(part.lower() in {".aws", ".ssh", ".gnupg"} for part in candidate.parts)


def _is_device_path(path: str) -> bool:
    return Path(path).is_relative_to(Path("/dev"))
