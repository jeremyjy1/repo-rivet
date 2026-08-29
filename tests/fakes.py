from collections.abc import Iterable
from typing import Any

from repo_rivet.llm.base import ModelRequestOptions, ModelResponse
from repo_rivet.tools.base import ToolCall, ToolResult


class FakeModelClient:
    def __init__(self, responses: Iterable[ModelResponse | Exception]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: ModelRequestOptions | None = None,
    ) -> ModelResponse:
        self.requests.append({"messages": messages, "tools": tools, "options": options})
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeToolRegistry:
    def __init__(self, results: Iterable[ToolResult]) -> None:
        self._results = iter(results)
        self.calls: list[ToolCall] = []

    def schemas(self) -> list[dict[str, Any]]:
        return []

    def modifies_workspace_files(self, tool_name: str) -> bool:
        return tool_name in {"edit_file", "write_file"}

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return next(self._results)
