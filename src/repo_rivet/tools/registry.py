"""Registration, schema generation, and dispatch for local tools."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from repo_rivet.approval.engine import ApprovalEngine
from repo_rivet.approval.models import ApprovalAction, Capability
from repo_rivet.editing.runtime import EditingRuntime
from repo_rivet.editing.tools import EditFileTool
from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import BaseTool, DecisionPolicy, ToolCall, ToolResult
from repo_rivet.tools.filesystem import (
    ListFilesTool,
    ReadFileTool,
    SearchTextTool,
    WriteFileTool,
)
from repo_rivet.tools.git import GitDiffTool
from repo_rivet.tools.meta import RecordDecisionTool
from repo_rivet.tools.shell import RunCommandTool
from repo_rivet.tools.verification import RegisterVerificationTool, RunVerificationTool
from repo_rivet.verification.runtime import VerificationRuntime

_STATE_CHANGING_CAPABILITIES = frozenset(
    {
        Capability.FILESYSTEM_WRITE,
        Capability.FILESYSTEM_DELETE,
        Capability.PROCESS_EXECUTE,
        Capability.NETWORK_ACCESS,
        Capability.GIT_WRITE,
        Capability.GIT_HISTORY_REWRITE,
    }
)

_WORKSPACE_FILE_CHANGE_CAPABILITIES = frozenset(
    {
        Capability.FILESYSTEM_WRITE,
        Capability.FILESYSTEM_DELETE,
    }
)


class ToolRegistry:
    """Own the available tools and dispatch validated model calls."""

    def __init__(
        self,
        tools: Iterable[BaseTool[Any]] = (),
        *,
        workspace: Path | None = None,
        approval_engine: ApprovalEngine | None = None,
        verification_runtime: VerificationRuntime | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}
        self.workspace = workspace
        self.approval_engine = approval_engine
        self.verification_runtime = verification_runtime
        for tool in tools:
            self.register(tool)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def register(self, tool: BaseTool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    def is_state_changing(self, tool_name: str) -> bool:
        """Return whether a registered tool declares any side-effect capability."""
        tool = self._tools.get(tool_name)
        return bool(tool and tool.capabilities & _STATE_CHANGING_CAPABILITIES)

    def modifies_workspace_files(self, tool_name: str) -> bool:
        """Return whether a tool declares workspace file mutation capabilities."""
        tool = self._tools.get(tool_name)
        return bool(tool and tool.capabilities & _WORKSPACE_FILE_CHANGE_CAPABILITIES)

    def decision_policy(self, tool_name: str) -> DecisionPolicy:
        """Return the declared source of auditable intent for a tool execution."""
        tool = self._tools.get(tool_name)
        return tool.decision_policy if tool is not None else DecisionPolicy.MUTATION

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self.names) or "none"
            return ToolResult(
                ok=False,
                output="",
                error=f"Unknown tool: {call.name}. Available tools: {available}",
            )
        try:
            validated = tool.validate_arguments(call.arguments)
            if isinstance(validated, ToolResult):
                return validated
            if self.approval_engine is None:
                return tool.execute_validated(validated)

            normalized_arguments = tool.approval_arguments(validated)
            if isinstance(normalized_arguments, ToolResult):
                return normalized_arguments
            outcome = self.approval_engine.authorize(
                tool_name=call.name,
                arguments=normalized_arguments,
                capabilities=tool.capabilities,
                session_id=self.approval_engine.session_id,
            )
            decision = outcome.decision
            if decision.action != ApprovalAction.ALLOW:
                error_code = (
                    "hard_policy_denied" if decision.source == "hard_policy" else "approval_denied"
                )
                error = decision.reason
                if decision.guidance:
                    error = f"{error}. User direction: {decision.guidance}"
                return ToolResult(
                    ok=False,
                    output="",
                    error=error,
                    error_code=error_code,
                    retryable=False,
                    metadata={
                        "approval_source": decision.source,
                        "approval_fingerprint": decision.request_fingerprint,
                        "approval_abort": decision.abort_agent,
                        "approval_guidance": decision.guidance,
                    },
                )

            stale_decision = self.approval_engine.revalidate(outcome)
            if stale_decision is not None:
                return ToolResult(
                    ok=False,
                    output="",
                    error=stale_decision.reason,
                    error_code=(
                        "approval_stale"
                        if stale_decision.source == "execution_revalidation"
                        else "hard_policy_denied"
                    ),
                    retryable=False,
                    metadata={"approval_source": stale_decision.source},
                )
            tool.approval_granted(validated, source=decision.source)
            self.approval_engine.record_execution_started(outcome)
            result = tool.execute_validated(validated)
            self.approval_engine.record_execution(
                outcome,
                ok=result.ok,
                metadata=result.metadata,
            )
            return result
        except Exception:  # pragma: no cover - defensive boundary around local integrations
            return ToolResult(ok=False, output="", error=f"Tool failed unexpectedly: {call.name}")


def create_default_registry(
    workspace: str | Path,
    *,
    approval_engine: ApprovalEngine | None = None,
    snapshot_dir: Path | None = None,
    event_logger: Any | None = None,
    initial_workspace_revision: int = 0,
) -> ToolRegistry:
    """Create the local workspace tools and the side-effect-free decision meta tool."""
    path_policy = WorkspacePathPolicy(workspace)
    command_policy = CommandPolicy()
    verification_runtime = VerificationRuntime(path_policy, command_policy)
    editing_runtime = EditingRuntime(
        path_policy,
        snapshot_dir=snapshot_dir,
        event_logger=event_logger,
        initial_workspace_revision=initial_workspace_revision,
    )
    return ToolRegistry(
        [
            RecordDecisionTool(),
            RegisterVerificationTool(),
            ListFilesTool(path_policy),
            SearchTextTool(path_policy, editing_runtime),
            ReadFileTool(path_policy, editing_runtime),
            WriteFileTool(path_policy, editing_runtime),
            EditFileTool(editing_runtime),
            RunCommandTool(path_policy, command_policy),
            RunVerificationTool(verification_runtime),
            GitDiffTool(path_policy),
        ],
        workspace=path_policy.workspace,
        approval_engine=approval_engine,
        verification_runtime=verification_runtime,
    )
