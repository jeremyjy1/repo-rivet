"""Mutable state for one explicit RepoRivet agent loop."""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from repo_rivet.llm.base import ModelResponse
from repo_rivet.tools.base import ToolCall, ToolResult

_FILE_MODIFICATION_TOOLS = frozenset({"replace_text", "write_file"})


@dataclass(slots=True)
class SessionState:
    """Track conversation progress and safety-relevant counters."""

    task: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    tool_call_count: int = 0
    modified_files: set[str] = field(default_factory=set)
    last_change_step: int | None = None
    last_verification_step: int | None = None
    last_verification_success: bool = False
    consecutive_failures: int = 0
    repeated_tool_calls: int = 0
    empty_model_responses: int = 0
    last_tool_signature: str | None = None
    recent_errors: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    interrupted: bool = False

    def record_model_response(self, response: ModelResponse) -> None:
        """Advance one model step and track unusable empty responses."""
        self.step_count += 1
        if response.tool_calls or (response.content and response.content.strip()):
            self.empty_model_responses = 0
        else:
            self.empty_model_responses += 1
        self.messages.append(response.as_assistant_message())

    def record_model_error(self, error: str) -> None:
        """Record an unusable model response and request a corrected response."""
        self.step_count += 1
        self.empty_model_responses += 1
        self.recent_errors.append(error)
        self.recent_errors[:] = self.recent_errors[-5:]
        self.messages.append(
            {
                "role": "system",
                "content": (
                    f"The previous model response was invalid: {error}. "
                    "Return valid text or a valid function tool call."
                ),
            }
        )

    def record_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        """Update counters and file-change state after one ordered tool call."""
        self.tool_call_count += 1
        signature = json.dumps(
            {"name": call.name, "arguments": call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if signature == self.last_tool_signature:
            self.repeated_tool_calls += 1
        else:
            self.last_tool_signature = signature
            self.repeated_tool_calls = 1

        if result.ok:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if result.error:
                self.recent_errors.append(result.error)
                self.recent_errors[:] = self.recent_errors[-5:]

        if result.ok and call.name in _FILE_MODIFICATION_TOOLS:
            path = call.arguments.get("path")
            if isinstance(path, str):
                self.modified_files.add(path)
            self.last_change_step = self.tool_call_count
            self.last_verification_success = False

        self.messages.append(result.as_tool_message(call.id))

    def state_summary(self) -> str:
        """Return a compact deterministic summary for the context manager."""
        modified = ", ".join(sorted(self.modified_files)) or "none"
        verification = (
            "passed"
            if self.last_verification_success
            else "required"
            if self.modified_files
            else "not required"
        )
        errors = " | ".join(self.recent_errors) or "none"
        return (
            f"Model steps: {self.step_count}. Tool calls: {self.tool_call_count}.\n"
            f"Modified files: {modified}.\n"
            f"Latest verification: {verification}.\n"
            f"Recent errors: {errors}."
        )
