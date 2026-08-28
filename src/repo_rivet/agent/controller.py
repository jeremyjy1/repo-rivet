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
from repo_rivet.reasoning.manager import ReasoningManager
from repo_rivet.reasoning.models import ReasoningEvent, ReasoningPhase
from repo_rivet.reasoning.validator import DecisionValidationError, validate_decision_for_actions
from repo_rivet.tools.base import ToolCall, ToolResult
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
        reasoning_manager: ReasoningManager | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.context_manager = context_manager or ContextManager()
        self.verifier = verifier or Verifier()
        self.termination_policy = termination_policy or TerminationPolicy()
        self.event_logger = event_logger
        self.memory_store = memory_store
        self.reasoning_manager = reasoning_manager or ReasoningManager()

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
        repaired_legacy_reflection = self._repair_legacy_protocol_reflection(memory)
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
            reflection_required=memory.reflection_required or bool(memory.invalidated_files),
        )
        self._log("session_start", task=state.task)
        if repaired_legacy_reflection:
            self._log(
                "legacy_decision_state_repaired",
                reason="trailing decision-validation rejections did not execute tools",
            )
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
                        tool_schemas=self._tool_schemas(),
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
                    terminal_result = self._process_tool_turn(
                        response.tool_calls,
                        state=state,
                        memory=memory,
                    )
                    if terminal_result is not None:
                        return terminal_result
                    continue

                if state.pending_decision is not None:
                    state.pending_decision = None
                    memory.working.pending_actions.clear()
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

    def _process_tool_turn(
        self,
        calls: list[ToolCall],
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        reasoning_calls = [call for call in calls if call.name == "record_decision"]
        action_calls = [call for call in calls if call.name != "record_decision"]
        mutating_calls = [call for call in action_calls if self._is_state_changing(call.name)]
        pending_decision = state.pending_decision
        state.pending_decision = None
        if reasoning_calls or action_calls:
            memory.working.pending_actions.clear()
        reasoning_event: ReasoningEvent | None = None
        reasoning_error: str | None = None

        if len(reasoning_calls) > 1:
            reasoning_error = "Only one record_decision call is allowed per model turn."
        elif reasoning_calls:
            if not self.reasoning_manager.config.enabled:
                reasoning_error = "Structured decision recording is disabled for this runtime."
            else:
                try:
                    reasoning_event = self.reasoning_manager.record(
                        reasoning_calls[0].arguments,
                        memory=memory,
                        step=state.step_count,
                    )
                except ValueError as error:
                    reasoning_error = str(error)

        decision_for_validation = reasoning_event
        if reasoning_event is None and mutating_calls:
            decision_for_validation = pending_decision

        validation_error = reasoning_error
        if mutating_calls and state.reflection_required:
            validation_error = (
                "The previous action failed or was denied. Record a reflection in a separate "
                "turn before declaring another state-changing action."
            )
        if validation_error is None:
            try:
                validate_decision_for_actions(
                    decision_for_validation,
                    action_calls,
                    mutating_calls=mutating_calls,
                    require_decision=any(self._requires_decision(call) for call in mutating_calls),
                )
            except DecisionValidationError as error:
                validation_error = str(error)

        if validation_error is not None:
            state.record_protocol_failure(validation_error)
        else:
            state.consecutive_protocol_failures = 0

        if reasoning_event is not None:
            self._log("reasoning", **reasoning_event.model_dump(mode="json"))
            self._save_memory(memory, state, status="running")

        reasoning_call_id = reasoning_calls[0].id if len(reasoning_calls) == 1 else None
        for call in calls:
            self._log(
                "tool_call",
                step=state.step_count,
                tool_call_id=call.id,
                name=call.name,
                arguments=call.arguments,
            )
            if call.name == "record_decision":
                if reasoning_event is not None and call.id == reasoning_call_id:
                    deferred_action = (
                        not action_calls
                        and reasoning_event.phase == ReasoningPhase.DECISION
                        and reasoning_event.next_action is not None
                        and not state.reflection_required
                    )
                    if deferred_action:
                        suffix = (
                            " It authorizes that matching state-changing tool only in the "
                            "immediately following model response."
                        )
                    elif not action_calls and reasoning_event.phase == ReasoningPhase.REFLECTION:
                        suffix = (
                            " The reflection gate is cleared; the next state-changing tool "
                            "still requires a matching decision."
                        )
                    elif not action_calls and state.reflection_required:
                        suffix = (
                            " A standalone reflection is still required before another "
                            "state-changing tool."
                        )
                    else:
                        suffix = ""
                    result = ToolResult(
                        ok=True,
                        output=(
                            f"Recorded {reasoning_event.phase.value} as "
                            f"{reasoning_event.event_id}.{suffix}"
                        ),
                        metadata={"evidence_ref": reasoning_event.event_id},
                    )
                else:
                    result = ToolResult(
                        ok=False,
                        output="",
                        error=reasoning_error or "Duplicate decision record.",
                        error_code="reasoning_invalid",
                        retryable=True,
                    )
                self._record_meta_result(call, result, state=state, memory=memory)
                continue

            if validation_error is not None:
                result = ToolResult(
                    ok=False,
                    output="",
                    error=validation_error,
                    error_code="decision_validation_failed",
                    retryable=True,
                )
                self._record_blocked_action(call, result, state=state, memory=memory)
                continue

            self._log(
                "action",
                step=state.step_count,
                tool_call_id=call.id,
                tool=call.name,
                argument_summary=self._action_summary(call),
            )
            result = self.tool_registry.execute(call)
            terminal_result = self._record_action_result(call, result, state=state, memory=memory)
            if terminal_result is not None:
                return terminal_result

        if reasoning_event is not None and not action_calls:
            if reasoning_event.phase in {ReasoningPhase.PLAN, ReasoningPhase.REFLECTION}:
                state.reasoning_only_turns += 1
            else:
                state.reasoning_only_turns = 0
            if reasoning_event.phase == ReasoningPhase.REFLECTION:
                state.reflection_required = False
                memory.reflection_required = False
            elif (
                reasoning_event.phase == ReasoningPhase.DECISION
                and reasoning_event.next_action is not None
                and not state.reflection_required
            ):
                state.pending_decision = reasoning_event
            if state.reasoning_only_turns > self.reasoning_manager.config.max_reflection_only_turns:
                prompt = (
                    "Too many reasoning-only turns. Take a concrete tool action now, provide a "
                    "final answer, or state why progress is impossible."
                )
                state.messages.append({"role": "system", "content": prompt})
                memory.messages.append(
                    Message.from_chat_message(state.messages[-1], step=state.tool_call_count)
                )
            self._save_memory(memory, state, status="running")
        elif action_calls:
            state.reasoning_only_turns = 0
        if validation_error is not None:
            reason = self.termination_policy.check(state)
            if reason:
                return self._finish(state, memory, status="stopped", reason=reason)
        return None

    def _record_meta_result(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> None:
        message = result.as_tool_message(call.id)
        state.messages.append(message)
        memory.append_tool_message(message, step=state.tool_call_count)
        self._log(
            "tool_result",
            tool_call_id=call.id,
            name=call.name,
            ok=result.ok,
            error=result.error,
            error_code=result.error_code,
            retryable=result.retryable,
            metadata=result.metadata,
        )
        self._save_memory(memory, state, status="running")

    def _record_blocked_action(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> None:
        """Audit a rejected model call without pretending the tool executed."""
        state.record_blocked_tool_result(call, result)
        memory.append_tool_message(result.as_tool_message(call.id), step=state.tool_call_count)
        self._log(
            "action_blocked",
            step=state.step_count,
            tool_call_id=call.id,
            tool=call.name,
            reason=result.error,
            error_code=result.error_code,
        )
        self._log(
            "tool_result",
            tool_call_id=call.id,
            name=call.name,
            ok=False,
            error=result.error,
            error_code=result.error_code,
            retryable=result.retryable,
            metadata=result.metadata,
            executed=False,
        )
        self._save_memory(memory, state, status="running")

    def _record_action_result(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        next_step = state.tool_call_count + 1
        output_ref = (
            self.memory_store.save_tool_output(call, result, step=next_step)
            if self.memory_store is not None
            else None
        )
        observation = self.reasoning_manager.observe(
            call,
            result,
            memory=memory,
            step=next_step,
            output_ref=output_ref,
        )
        result = self.reasoning_manager.result_with_evidence(result, observation)
        if not result.ok:
            state.reflection_required = True
            memory.reflection_required = True
        state.record_tool_result(call, result)
        self.verifier.observe(state, call, result)
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
        self._log("observation", **observation.model_dump(mode="json"))
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
        return None

    def _tool_schemas(self) -> list[dict[str, Any]]:
        schemas = self.tool_registry.schemas()
        if self.reasoning_manager.config.enabled:
            return schemas
        return [
            schema
            for schema in schemas
            if schema.get("function", {}).get("name") != "record_decision"
        ]

    def _is_state_changing(self, tool_name: str) -> bool:
        checker = getattr(self.tool_registry, "is_state_changing", None)
        if callable(checker):
            return bool(checker(tool_name))
        return tool_name in {"replace_text", "run_command", "write_file"}

    def _requires_decision(self, call: ToolCall) -> bool:
        config = self.reasoning_manager.config
        if not config.enabled:
            return False
        if call.name == "run_command":
            return config.require_for_commands
        return config.require_for_mutating_tools

    @staticmethod
    def _repair_legacy_protocol_reflection(memory: MemoryState) -> bool:
        """Clear reflection state created solely by pre-fix protocol observations."""
        if not memory.reflection_required or memory.invalidated_files:
            return False
        saw_legacy_protocol_observation = False
        for event in reversed(memory.observation_events):
            if "decision_validation_failed" in event.result_summary:
                saw_legacy_protocol_observation = True
                continue
            if not event.ok:
                return False
            break
        if not saw_legacy_protocol_observation:
            return False
        memory.reflection_required = False
        return True

    @staticmethod
    def _action_summary(call: ToolCall) -> str:
        path = call.arguments.get("path") or call.arguments.get("cwd")
        if isinstance(path, str):
            return path
        if call.name == "run_command":
            command = call.arguments.get("command")
            if isinstance(command, str):
                return command.split(maxsplit=1)[0]
        return ""

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
