"""Registration, schema generation, and dispatch for local tools."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from repo_rivet.approval.engine import ApprovalEngine
from repo_rivet.approval.models import ApprovalAction
from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import BaseTool, ToolCall, ToolResult
from repo_rivet.tools.filesystem import (
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
    SearchTextTool,
    WriteFileTool,
)
from repo_rivet.tools.git import GitDiffTool
from repo_rivet.tools.shell import RunCommandTool


class ToolRegistry:
    """Own the available tools and dispatch validated model calls."""

    def __init__(
        self,
        tools: Iterable[BaseTool[Any]] = (),
        *,
        workspace: Path | None = None,
        approval_engine: ApprovalEngine | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}
        self.workspace = workspace
        self.approval_engine = approval_engine
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

            normalized_arguments = validated.model_dump(mode="json")
            outcome = self.approval_engine.authorize(
                tool_name=call.name,
                arguments=normalized_arguments,
                capabilities=tool.capabilities,
                session_id=self.approval_engine.session_id,
            )
            decision = outcome.decision
            if decision.action != ApprovalAction.ALLOW:
                error_code = (
                    "hard_policy_denied"
                    if decision.source == "hard_policy"
                    else "approval_denied"
                )
                return ToolResult(
                    ok=False,
                    output="",
                    error=decision.reason,
                    error_code=error_code,
                    retryable=False,
                    metadata={
                        "approval_source": decision.source,
                        "approval_fingerprint": decision.request_fingerprint,
                        "approval_abort": decision.abort_agent,
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
) -> ToolRegistry:
    """Create the seven minimum tools for one confined workspace."""
    path_policy = WorkspacePathPolicy(workspace)
    command_policy = CommandPolicy()
    return ToolRegistry(
        [
            ListFilesTool(path_policy),
            SearchTextTool(path_policy),
            ReadFileTool(path_policy),
            WriteFileTool(path_policy),
            ReplaceTextTool(path_policy),
            RunCommandTool(path_policy, command_policy),
            GitDiffTool(path_policy),
        ],
        workspace=path_policy.workspace,
        approval_engine=approval_engine,
    )
