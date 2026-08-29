"""Deterministic limits for the explicit agent loop."""

import time
from dataclasses import dataclass

from repo_rivet.agent.state import SessionState


@dataclass(frozen=True, slots=True)
class TerminationConfig:
    max_steps: int = 30
    max_seconds: float = 600
    max_consecutive_failures: int = 5
    max_consecutive_protocol_failures: int = 5
    max_repeated_tool_calls: int = 3
    max_empty_model_responses: int = 3
    max_consecutive_length_responses: int = 3

    def __post_init__(self) -> None:
        values = {
            "max_steps": self.max_steps,
            "max_seconds": self.max_seconds,
            "max_consecutive_failures": self.max_consecutive_failures,
            "max_consecutive_protocol_failures": self.max_consecutive_protocol_failures,
            "max_repeated_tool_calls": self.max_repeated_tool_calls,
            "max_empty_model_responses": self.max_empty_model_responses,
            "max_consecutive_length_responses": self.max_consecutive_length_responses,
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError("Termination limits must be positive")


class TerminationPolicy:
    """Return an explicit stop reason whenever a configured limit is reached."""

    def __init__(self, config: TerminationConfig | None = None) -> None:
        self.config = config or TerminationConfig()

    def step_limit(self, state: SessionState) -> int:
        """Return the current progress-checkpoint boundary."""
        return state.step_limit or self.config.max_steps

    def step_checkpoint_reason(self, state: SessionState) -> str:
        return f"maximum agent step checkpoint reached ({self.step_limit(state)})"

    def check(
        self,
        state: SessionState,
        *,
        now: float | None = None,
        include_step_limit: bool = True,
    ) -> str | None:
        if state.interrupted:
            return "interrupted by user"
        step_limit = self.step_limit(state)
        if include_step_limit and state.step_count >= step_limit:
            return self.step_checkpoint_reason(state)

        current_time = time.monotonic() if now is None else now
        if current_time - state.started_at >= self.config.max_seconds:
            return f"maximum runtime reached ({self.config.max_seconds:g} seconds)"
        if state.consecutive_failures >= self.config.max_consecutive_failures:
            return (
                "maximum consecutive tool failures reached "
                f"({self.config.max_consecutive_failures})"
            )
        if state.consecutive_protocol_failures >= self.config.max_consecutive_protocol_failures:
            return (
                "maximum consecutive decision protocol failures reached "
                f"({self.config.max_consecutive_protocol_failures})"
            )
        if state.repeated_tool_calls >= self.config.max_repeated_tool_calls:
            return (
                "repeated identical tool call limit reached "
                f"({self.config.max_repeated_tool_calls})"
            )
        if state.empty_model_responses >= self.config.max_empty_model_responses:
            return (
                f"maximum empty model responses reached ({self.config.max_empty_model_responses})"
            )
        if state.consecutive_length_responses >= self.config.max_consecutive_length_responses:
            return (
                "maximum consecutive length-limited model responses reached "
                f"({self.config.max_consecutive_length_responses})"
            )
        return None
