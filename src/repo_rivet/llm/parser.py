"""Normalize OpenAI-compatible Chat Completions responses."""

import json
from typing import Any

from repo_rivet.llm.base import ModelResponse
from repo_rivet.tools.base import ToolCall


class ResponseParseError(ValueError):
    """Raised when a provider response cannot be represented safely."""


class ResponseParser:
    """Parse text and native function calls from a provider response."""

    def parse(self, response: Any) -> ModelResponse:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ResponseParseError("Model response contains no choices")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise ResponseParseError("Model response choice contains no message")

        content = getattr(message, "content", None)
        if content is not None and not isinstance(content, str):
            raise ResponseParseError("Model response content is not text")
        reasoning_content = self._provider_text(message, "reasoning_content")

        tool_calls = [
            self._parse_tool_call(item) for item in getattr(message, "tool_calls", None) or []
        ]
        return ModelResponse(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    @staticmethod
    def _provider_text(message: Any, field_name: str) -> str | None:
        value = getattr(message, field_name, None)
        if value is None:
            model_extra = getattr(message, "model_extra", None)
            if isinstance(model_extra, dict):
                value = model_extra.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ResponseParseError(f"Model response {field_name} is not text")
        return value

    @staticmethod
    def _parse_tool_call(raw_call: Any) -> ToolCall:
        call_type = getattr(raw_call, "type", "function")
        if call_type != "function":
            raise ResponseParseError(f"Unsupported tool call type: {call_type}")

        call_id = getattr(raw_call, "id", None)
        function = getattr(raw_call, "function", None)
        name = getattr(function, "name", None)
        raw_arguments = getattr(function, "arguments", None)
        if not isinstance(call_id, str) or not call_id:
            raise ResponseParseError("Tool call is missing an id")
        if not isinstance(name, str) or not name:
            raise ResponseParseError("Tool call is missing a function name")

        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise ResponseParseError(
                    f"Tool call {name} contains invalid JSON arguments: {error.msg}"
                ) from None
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise ResponseParseError(f"Tool call {name} has unsupported arguments")

        if not isinstance(arguments, dict):
            raise ResponseParseError(f"Tool call {name} arguments must be a JSON object")
        return ToolCall(id=call_id, name=name, arguments=arguments)
