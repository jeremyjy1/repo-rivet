from collections.abc import Iterable
from typing import Any

from repo_rivet.llm.base import ModelResponse
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
    ) -> ModelResponse:
        self.requests.append({"messages": messages, "tools": tools})
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

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return next(self._results)
