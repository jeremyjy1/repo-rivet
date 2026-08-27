"""Explicit single-agent model/tool loop."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from repo_rivet.agent.state import SessionState
from repo_rivet.agent.termination import TerminationPolicy
from repo_rivet.agent.verifier import Verifier
from repo_rivet.context.manager import ContextManager
from repo_rivet.llm.base import ModelClient
from repo_rivet.llm.parser import ResponseParseError
from repo_rivet.tools.registry import ToolRegistry


class EventSink(Protocol):
    """Minimal logging interface accepted by the controller."""

    def log(self, event_type: str, **data: Any) -> None:
        """Record a structured agent event."""
        ...


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Final outcome returned to the CLI for every terminal path."""

    status: Literal["success", "stopped", "error"]
    summary: str
    reason: str | None
    modified_files: tuple[str, ...]
    step_count: int
    tool_call_count: int
    verification_success: bool


class AgentController:
    """Drive model decisions and sequential local tool execution."""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
        context_manager: ContextManager | None = None,
        verifier: Verifier | None = None,
        termination_policy: TerminationPolicy | None = None,
        event_logger: EventSink | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.context_manager = context_manager or ContextManager()
        self.verifier = verifier or Verifier()
        self.termination_policy = termination_policy or TerminationPolicy()
        self.event_logger = event_logger

    def run(self, task: str) -> AgentResult:
        """Run until verified success, a deterministic stop, or a model API error."""
        if not task.strip():
            raise ValueError("Task must not be empty")

        state = SessionState(task=task.strip())
        self._log("session_start", task=state.task)
        try:
            while True:
                reason = self.termination_policy.check(state)
                if reason:
                    return self._finish(state, status="stopped", reason=reason)

                messages = self.context_manager.build(
                    task=state.task,
                    history=state.messages,
                    state_summary=state.state_summary(),
                    remaining_steps=self.termination_policy.config.max_steps - state.step_count,
                )
                self._log("model_call", step=state.step_count + 1, message_count=len(messages))
                try:
                    response = self.model_client.complete(
                        messages=messages,
                        tools=self.tool_registry.schemas(),
                    )
                except ResponseParseError as error:
                    state.record_model_error(str(error))
                    self._log("model_response_invalid", step=state.step_count, error=str(error))
                    continue
                except Exception as error:
                    return self._finish(
                        state,
                        status="error",
                        reason=f"model API failed after retries: {type(error).__name__}",
                    )

                state.record_model_response(response)
                self._log(
                    "model_response",
                    step=state.step_count,
                    finish_reason=response.finish_reason,
                    content_length=len(response.content or ""),
                    tools=[call.name for call in response.tool_calls],
                )
                if response.tool_calls:
                    for call in response.tool_calls:
                        self._log(
                            "tool_call",
                            step=state.step_count,
                            tool_call_id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                        )
                        result = self.tool_registry.execute(call)
                        state.record_tool_result(call, result)
                        self.verifier.observe(state, call, result)
                        self._log(
                            "tool_result",
                            tool_call_id=call.id,
                            name=call.name,
                            ok=result.ok,
                            error=result.error,
                            metadata=result.metadata,
                        )

                        reason = self.termination_policy.check(state)
                        if reason:
                            return self._finish(state, status="stopped", reason=reason)
                    continue

                if response.content and response.content.strip():
                    if self.verifier.can_finish(state):
                        return self._finish(
                            state,
                            status="success",
                            summary=response.content.strip(),
                        )
                    state.messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Files changed after the latest successful verification. "
                                "Run an appropriate test, build, lint, syntax check, or git_diff "
                                "before finishing."
                            ),
                        }
                    )
        except KeyboardInterrupt:
            state.interrupted = True
            return self._finish(state, status="stopped", reason="interrupted by user")

    def _finish(
        self,
        state: SessionState,
        *,
        status: Literal["success", "stopped", "error"],
        reason: str | None = None,
        summary: str = "",
    ) -> AgentResult:
        result = self._result(state, status=status, reason=reason, summary=summary)
        self._log(
            "session_end",
            status=result.status,
            reason=result.reason,
            modified_files=result.modified_files,
            step_count=result.step_count,
            tool_call_count=result.tool_call_count,
            verification_success=result.verification_success,
        )
        return result

    def _log(self, event_type: str, **data: Any) -> None:
        if self.event_logger is not None:
            self.event_logger.log(event_type, **data)

    @staticmethod
    def _result(
        state: SessionState,
        *,
        status: Literal["success", "stopped", "error"],
        reason: str | None = None,
        summary: str = "",
    ) -> AgentResult:
        if not summary:
            modified = ", ".join(sorted(state.modified_files)) or "none"
            errors = " | ".join(state.recent_errors) or "none"
            summary = f"Modified files: {modified}. Recent errors: {errors}."
        return AgentResult(
            status=status,
            summary=summary,
            reason=reason,
            modified_files=tuple(sorted(state.modified_files)),
            step_count=state.step_count,
            tool_call_count=state.tool_call_count,
            verification_success=state.last_verification_success,
        )
