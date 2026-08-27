"""Registration, schema generation, and dispatch for local tools."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

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

    def __init__(self, tools: Iterable[BaseTool[Any]] = ()) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}
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
            return tool.execute(call.arguments)
        except Exception:  # pragma: no cover - defensive boundary around local integrations
            return ToolResult(ok=False, output="", error=f"Tool failed unexpectedly: {call.name}")


def create_default_registry(workspace: str | Path) -> ToolRegistry:
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
        ]
    )
