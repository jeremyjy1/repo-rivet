"""Provider-independent model interfaces and response types."""

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from repo_rivet.tools.base import ToolCall


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A normalized response returned by any model adapter."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    def as_assistant_message(self) -> dict[str, Any]:
        """Serialize the normalized response into conversation history."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
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
    ) -> ModelResponse:
        """Return the model's next message and optional tool calls."""
        ...
