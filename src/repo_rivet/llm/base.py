"""Provider-independent model interfaces and response types."""

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from repo_rivet.tools.base import ToolCall


class ModelContextLengthError(RuntimeError):
    """A provider rejected the request because its context was too large."""


@dataclass(frozen=True, slots=True)
class ModelRequestOptions:
    """Per-request provider controls used for bounded recovery."""

    reasoning_effort: Literal["low", "high", "max"] | None = None
    thinking_enabled: bool | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A normalized response returned by any model adapter."""

    content: str | None = None
    reasoning_content: str | None = field(default=None, repr=False)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_thinking_disabled: bool = False
    reasoning_context_restart_required: bool = False

    def as_assistant_message(self) -> dict[str, Any]:
        """Serialize the normalized response into conversation history."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        return message


class ModelClient(Protocol):
    """Synchronous interface required by the agent controller."""

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: ModelRequestOptions | None = None,
    ) -> ModelResponse:
        """Return the model's next message and optional tool calls."""
        ...
