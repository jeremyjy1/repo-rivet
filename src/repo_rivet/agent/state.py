"""Mutable state for one explicit RepoRivet agent loop."""

import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from repo_rivet.llm.base import ModelResponse
from repo_rivet.reasoning.models import ReasoningEvent
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.verification.models import (
    FinalAssessment,
    VerificationPlan,
    VerificationResult,
    VerificationStatus,
)

_FILE_MODIFICATION_TOOLS = frozenset({"replace_text", "write_file"})


class AgentStatus(StrEnum):
    RUNNING = "running"
    VERIFYING = "verifying"
    FINALIZING = "finalizing"
    AWAITING_VERIFICATION_PLAN = "awaiting_verification_plan"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(slots=True)
class SessionState:
    """Track conversation progress and safety-relevant counters."""

    task: str
    status: AgentStatus = AgentStatus.RUNNING
    messages: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    tool_call_count: int = 0
    initial_tool_call_count: int = 0
    modified_files: set[str] = field(default_factory=set)
    workspace_revision: int = 0
    verification_plan: VerificationPlan | None = None
    verification_results: dict[str, VerificationResult] = field(default_factory=dict)
    verification_plan_recovery_attempts: int = 0
    candidate_final_assessment: FinalAssessment | None = None
    consecutive_failures: int = 0
    repeated_tool_calls: int = 0
    empty_model_responses: int = 0
    consecutive_length_responses: int = 0
    provider_reasoning_detected: bool = False
    force_thinking_disabled: bool = False
    consecutive_protocol_failures: int = 0
    last_tool_signature: str | None = None
    recent_errors: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    interrupted: bool = False
    reasoning_only_turns: int = 0
    reflection_required: bool = False
    pending_decision: ReasoningEvent | None = None

    def record_model_response(self, response: ModelResponse) -> None:
        """Advance one model step and track unusable empty responses."""
        self.step_count += 1
        if response.finish_reason == "length" and not response.tool_calls:
            self.consecutive_length_responses += 1
        else:
            self.consecutive_length_responses = 0
        if (
            response.finish_reason == "length"
            or response.tool_calls
            or (response.content and response.content.strip())
        ):
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
        self._record_tool_outcome(call, result)
        self.messages.append(result.as_tool_message(call.id))

    def record_automatic_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        """Track a controller-scheduled tool without inventing an assistant Tool Call."""
        self._record_tool_outcome(call, result)

    def _record_tool_outcome(self, call: ToolCall, result: ToolResult) -> None:
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
            self.workspace_revision += 1
            for check_id, verification in list(self.verification_results.items()):
                if verification.status != VerificationStatus.STALE:
                    self.verification_results[check_id] = verification.model_copy(
                        update={"status": VerificationStatus.STALE}
                    )

    def record_verification_result(self, result: VerificationResult) -> None:
        self.verification_results[result.check_id] = result

    def record_blocked_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        """Pair a rejected model call without counting it as an executed tool."""
        self.messages.append(result.as_tool_message(call.id))

    def record_protocol_failure(self, error: str) -> None:
        """Track one invalid model turn independently from executor failures."""
        self.consecutive_protocol_failures += 1
        self.recent_errors.append(error)
        self.recent_errors[:] = self.recent_errors[-5:]

    def state_summary(self) -> str:
        """Return a compact deterministic summary for the context manager."""
        modified = ", ".join(sorted(self.modified_files)) or "none"
        verification = "not required"
        if self.modified_files:
            if self.verification_plan is None:
                verification = "plan required"
            else:
                statuses = {
                    check.check_id: self.verification_results.get(check.check_id)
                    for check in self.verification_plan.checks
                    if check.required
                }
                verification = (
                    ", ".join(
                        f"{check_id}={result.status.value if result else 'pending'}"
                        for check_id, result in statuses.items()
                    )
                    or "no required checks"
                )
        errors = " | ".join(self.recent_errors) or "none"
        pending_decision = (
            self.pending_decision.next_action.tool_name
            if self.pending_decision is not None and self.pending_decision.next_action is not None
            else "none"
        )
        return (
            f"Agent status: {self.status.value}.\n"
            f"Model steps: {self.step_count}. Tool calls: {self.tool_call_count}.\n"
            f"Modified files: {modified}.\n"
            f"Workspace revision: {self.workspace_revision}.\n"
            f"Verification checks: {verification}.\n"
            f"One-shot pending decision: {pending_decision}.\n"
            f"Recent errors: {errors}."
        )
