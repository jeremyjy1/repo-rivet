"""Shared data structures and interfaces for local tool calls."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from repo_rivet.approval.models import Capability


class DecisionPolicy(StrEnum):
    """How a state-changing tool obtains an auditable execution intent."""

    MUTATION = "mutation"
    COMMAND = "command"
    REGISTERED_PLAN = "registered_plan"


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
    raw_output: str | None = None
    error_code: str | None = None
    retryable: bool | None = None

    def as_tool_message(self, tool_call_id: str) -> dict[str, Any]:
        """Serialize this result for a Chat Completions tool message."""
        content = json.dumps(
            {
                "ok": self.ok,
                "output": self.output,
                "error": self.error,
                "error_code": self.error_code,
                "retryable": self.retryable,
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
    capabilities: ClassVar[frozenset[Capability]] = frozenset()
    decision_policy: ClassVar[DecisionPolicy] = DecisionPolicy.MUTATION

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
        validated = self.validate_arguments(arguments)
        if isinstance(validated, ToolResult):
            return validated
        return self.execute_validated(validated)

    def validate_arguments(self, arguments: dict[str, Any]) -> ArgumentsT | ToolResult:
        """Validate without producing side effects so approval can run next."""
        try:
            validated = self.arguments_type.model_validate(arguments)
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()
            )
            return ToolResult(
                ok=False,
                output="",
                error=f"Invalid arguments: {details}",
                error_code="invalid_arguments",
                retryable=False,
            )
        return cast(ArgumentsT, validated)

    def execute_validated(self, arguments: ArgumentsT) -> ToolResult:
        """Invoke the operation after validation and approval."""
        try:
            return self.run(arguments)
        except UnicodeError as error:
            return ToolResult(
                ok=False,
                output="",
                error=str(error),
                error_code="encoding_error",
                retryable=False,
            )
        except OSError as error:
            return ToolResult(
                ok=False,
                output="",
                error=str(error),
                error_code="io_error",
                retryable=True,
            )
        except ValueError as error:
            return ToolResult(
                ok=False,
                output="",
                error=str(error),
                error_code=str(getattr(error, "code", "tool_error")),
                retryable=bool(getattr(error, "retryable", True)),
                metadata=getattr(error, "metadata", None),
            )

    def approval_arguments(self, arguments: ArgumentsT) -> dict[str, Any] | ToolResult:
        """Return the concrete request that the approval layer must evaluate."""
        return arguments.model_dump(mode="json")

    def approval_granted(self, arguments: ArgumentsT, *, source: str) -> None:
        """Observe a completed approval before local execution begins."""
        return None

    @abstractmethod
    def run(self, arguments: ArgumentsT) -> ToolResult:
        """Execute the tool with validated arguments."""
