"""Narrow deterministic rules for obviously harmless typed tools."""

from repo_rivet.approval.models import ApprovalRequest, Capability

_SAFE_TOOLS = frozenset({"list_files", "search_text", "git_diff"})


def is_obviously_safe(request: ApprovalRequest) -> bool:
    forbidden = {
        Capability.FILESYSTEM_WRITE,
        Capability.FILESYSTEM_DELETE,
        Capability.PROCESS_EXECUTE,
        Capability.NETWORK_ACCESS,
        Capability.SECRET_READ,
        Capability.OUTSIDE_WORKSPACE,
        Capability.PRIVILEGE_ESCALATION,
    }
    if request.assessment.capabilities & forbidden:
        return False
    return request.tool_name in _SAFE_TOOLS or request.tool_name == "read_file"
