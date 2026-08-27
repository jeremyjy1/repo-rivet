"""Explicit single-agent model/tool loop."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from repo_rivet.agent.state import SessionState
from repo_rivet.agent.termination import TerminationPolicy
from repo_rivet.agent.verifier import Verifier
from repo_rivet.context.manager import (
    SYSTEM_PROMPT,
    ContextBudgetExceededError,
    ContextManager,
)
from repo_rivet.llm.base import ModelClient, ModelContextLengthError, ModelResponse
from repo_rivet.llm.parser import ResponseParseError
from repo_rivet.memory.models import MemoryState, Message
from repo_rivet.memory.store import MemoryStore
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
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.context_manager = context_manager or ContextManager()
        self.verifier = verifier or Verifier()
        self.termination_policy = termination_policy or TerminationPolicy()
        self.event_logger = event_logger
        self.memory_store = memory_store

    def run(
        self,
        task: str,
        *,
        memory: MemoryState | None = None,
    ) -> AgentResult:
        """Run until verified success, a deterministic stop, or a model API error."""
        if not task.strip():
            raise ValueError("Task must not be empty")

        workspace = getattr(self.tool_registry, "workspace", None) or Path.cwd().resolve()
        memory = memory or MemoryState(session_id=f"memory-{uuid4().hex[:8]}")
        memory.start_task(
            task=task,
            workspace=str(workspace),
            system_prompt=SYSTEM_PROMPT,
            safety_rules=[
                "All file operations must stay inside the configured workspace.",
                "Commands run without a shell and obvious destructive commands are blocked.",
                "Denied tool requests must not be repeated unchanged.",
                "Never expose API keys, tokens, passwords, or local configuration contents.",
            ],
            completion_rules=[
                "Inspect relevant files before editing.",
                "After file changes, complete a successful verification before finishing.",
                "Report modified files, verification, and unresolved errors explicitly.",
            ],
            max_steps=self.termination_policy.config.max_steps,
        )
        approval_engine = getattr(self.tool_registry, "approval_engine", None)
        if approval_engine is not None:
            approval_engine.sync_memory_rule()
        state = SessionState(
            task=task.strip(),
            tool_call_count=memory.tool_event_step,
            initial_tool_call_count=memory.tool_event_step,
            modified_files=set(memory.modified_files),
            last_change_step=memory.last_file_change_step,
            last_verification_step=memory.last_verification_step,
            last_verification_success=memory.last_verification_success,
        )
        self._log("session_start", task=state.task)
        self._save_memory(memory, state, status="running")
        try:
            while True:
                reason = self.termination_policy.check(state)
                if reason:
                    return self._finish(state, memory, status="stopped", reason=reason)

                try:
                    response = self._complete_with_context_recovery(
                        state=state,
                        memory=memory,
                        tool_schemas=self.tool_registry.schemas(),
                    )
                except ResponseParseError as error:
                    state.record_model_error(str(error))
                    memory.messages.append(
                        Message.from_chat_message(state.messages[-1], step=state.tool_call_count)
                    )
                    self._log("model_response_invalid", step=state.step_count, error=str(error))
                    self._save_memory(memory, state, status="running")
                    continue
                except AgentContextOverflowError as error:
                    return self._finish(
                        state,
                        memory,
                        status="stopped",
                        reason=str(error),
                    )
                except Exception as error:
                    return self._finish(
                        state,
                        memory,
                        status="error",
                        reason=f"model API failed after retries: {type(error).__name__}",
                    )

                state.record_model_response(response)
                memory.total_input_tokens += (
                    response.input_tokens
                    if response.input_tokens is not None
                    else self.context_manager.last_request_tokens
                )
                memory.total_output_tokens += (
                    response.output_tokens
                    if response.output_tokens is not None
                    else self.context_manager.count_message(response.as_assistant_message())
                )
                memory.append_assistant(
                    response.as_assistant_message(),
                    step=state.tool_call_count,
                )
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
                        output_ref = (
                            self.memory_store.save_tool_output(
                                call,
                                result,
                                step=state.tool_call_count,
                            )
                            if self.memory_store is not None
                            else None
                        )
                        memory.record_tool_result(
                            call,
                            result,
                            step=state.tool_call_count,
                            full_output_path=output_ref,
                        )
                        memory.modified_files = set(state.modified_files)
                        memory.last_file_change_step = state.last_change_step
                        memory.last_verification_step = state.last_verification_step
                        memory.last_verification_success = state.last_verification_success
                        self._log(
                            "tool_result",
                            tool_call_id=call.id,
                            name=call.name,
                            ok=result.ok,
                            error=result.error,
                            error_code=result.error_code,
                            retryable=result.retryable,
                            metadata=result.metadata,
                            output_ref=output_ref,
                        )
                        self._save_memory(memory, state, status="running")

                        if result.metadata and result.metadata.get("approval_abort"):
                            return self._finish(
                                state,
                                memory,
                                status="stopped",
                                reason="agent aborted by user during tool approval",
                            )

                        reason = self.termination_policy.check(state)
                        if reason:
                            return self._finish(state, memory, status="stopped", reason=reason)
                    continue

                if response.content and response.content.strip():
                    if self.verifier.can_finish(state):
                        return self._finish(
                            state,
                            memory,
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
                    memory.messages.append(
                        Message.from_chat_message(state.messages[-1], step=state.tool_call_count)
                    )
                    self._save_memory(memory, state, status="running")
        except KeyboardInterrupt:
            state.interrupted = True
            return self._finish(state, memory, status="stopped", reason="interrupted by user")

    def _complete_with_context_recovery(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
        tool_schemas: list[dict[str, Any]],
    ) -> ModelResponse:
        max_retries = memory.config.max_context_overflow_retries
        last_reason = "context remained above the safe prompt budget"
        for attempt in range(max_retries + 1):
            if attempt:
                removed = self.context_manager.compact_for_recovery(
                    memory,
                    recovery_level=attempt,
                )
                self._log(
                    "context_recovery_compaction",
                    attempt=attempt,
                    removed_messages=removed,
                    remaining_messages=len(memory.messages),
                )
            try:
                messages = self.context_manager.build(
                    memory=memory,
                    state_summary=state.state_summary(),
                    remaining_steps=self.termination_policy.config.max_steps - state.step_count,
                    tools=tool_schemas,
                )
            except ContextBudgetExceededError as error:
                last_reason = str(error)
                self._log(
                    "context_preflight_overflow",
                    attempt=attempt,
                    reason=last_reason,
                )
                continue

            estimate = self.context_manager.last_estimate
            self._log(
                "model_call",
                step=state.step_count + 1,
                message_count=len(messages),
                context_recovery_attempt=attempt,
                raw_estimated_prompt_tokens=estimate.raw,
                effective_estimated_prompt_tokens=estimate.effective,
            )
            try:
                response = self.model_client.complete(messages=messages, tools=tool_schemas)
            except ModelContextLengthError:
                memory.context_overflow_count += 1
                self.context_manager.observe_overflow()
                last_reason = "provider rejected the request as exceeding its context limit"
                self._log(
                    "context_provider_overflow",
                    attempt=attempt,
                    raw_estimated_prompt_tokens=estimate.raw,
                    effective_estimated_prompt_tokens=estimate.effective,
                )
                continue

            if response.input_tokens is not None:
                self.context_manager.observe_usage(response.input_tokens)
            self._log(
                "model_usage",
                model=getattr(self.context_manager.token_manager, "model", "unknown"),
                raw_estimated_prompt_tokens=estimate.raw,
                effective_estimated_prompt_tokens=estimate.effective,
                actual_prompt_tokens=response.input_tokens,
                completion_tokens=response.output_tokens,
                correction_factor=estimate.correction_factor,
                context_limit=self.context_manager.token_manager.config.context_limit,
                prompt_budget=self.context_manager.token_manager.config.prompt_budget,
                tool_schema_estimate=(
                    self.context_manager.token_manager.state.tool_schema_estimate
                ),
                fixed_prompt_estimate=(
                    self.context_manager.token_manager.state.fixed_prompt_estimate
                ),
            )
            return response

        raise AgentContextOverflowError(
            f"Context could not be made safe after {max_retries} compression retries: {last_reason}"
        )

    def _finish(
        self,
        state: SessionState,
        memory: MemoryState,
        *,
        status: Literal["success", "stopped", "error"],
        reason: str | None = None,
        summary: str = "",
    ) -> AgentResult:
        result = self._result(state, status=status, reason=reason, summary=summary)
        memory_status = "completed" if status == "success" else status
        self._save_memory(memory, state, status=memory_status)
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

    def _save_memory(
        self,
        memory: MemoryState,
        state: SessionState,
        *,
        status: str,
    ) -> None:
        memory.status = status
        if self.memory_store is not None:
            self.memory_store.save_state(
                memory,
                agent_step=state.step_count,
                tool_call_count=state.tool_call_count,
                status=status,
            )

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
            tool_call_count=state.tool_call_count - state.initial_tool_call_count,
            verification_success=state.last_verification_success,
        )


class AgentContextOverflowError(RuntimeError):
    """Bounded context recovery was exhausted without a safe provider request."""
