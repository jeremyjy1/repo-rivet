"""Shared data structures and interfaces for local tool calls."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A validated request from the model to invoke a local tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The normalized result of a local tool invocation."""

    ok: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def as_tool_message(self, tool_call_id: str) -> dict[str, Any]:
        """Serialize this result for a Chat Completions tool message."""
        content = json.dumps(
            {
                "ok": self.ok,
                "output": self.output,
                "error": self.error,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
        )
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class ToolArguments(BaseModel):
    """Base class that rejects arguments not declared by a tool."""

    model_config = ConfigDict(extra="forbid")


class BaseTool[ArgumentsT: ToolArguments](ABC):
    """Validate model arguments before invoking a local operation."""

    name: ClassVar[str]
    description: ClassVar[str]
    arguments_type: ClassVar[type[ToolArguments]]

    @property
    def schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_type.model_json_schema(),
            },
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate arguments and normalize expected execution errors."""
        try:
            validated = self.arguments_type.model_validate(arguments)
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()
            )
            return ToolResult(ok=False, output="", error=f"Invalid arguments: {details}")

        try:
            return self.run(validated)  # type: ignore[arg-type]
        except (OSError, UnicodeError, ValueError) as error:
            return ToolResult(ok=False, output="", error=str(error))

    @abstractmethod
    def run(self, arguments: ArgumentsT) -> ToolResult:
        """Execute the tool with validated arguments."""
