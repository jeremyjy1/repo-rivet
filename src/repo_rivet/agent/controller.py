"""Explicit single-agent model/tool/verification loop."""

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from repo_rivet.agent.state import AgentStatus, SessionState
from repo_rivet.agent.termination import TerminationPolicy
from repo_rivet.agent.verifier import Verifier
from repo_rivet.context.manager import (
    SYSTEM_PROMPT,
    ContextBudgetExceededError,
    ContextManager,
)
from repo_rivet.llm.base import (
    ModelClient,
    ModelContextLengthError,
    ModelRequestOptions,
    ModelResponse,
)
from repo_rivet.llm.openai_compatible import ModelRequestError
from repo_rivet.llm.parser import ResponseParseError
from repo_rivet.llm.protocol import contains_embedded_tool_protocol
from repo_rivet.memory.models import MemoryState, Message
from repo_rivet.memory.store import MemoryStore
from repo_rivet.reasoning.manager import ReasoningManager
from repo_rivet.reasoning.models import ReasoningEvent, ReasoningPhase
from repo_rivet.reasoning.validator import DecisionValidationError, validate_decision_for_actions
from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.tools.registry import ToolRegistry
from repo_rivet.verification.models import (
    FINAL_ASSESSMENT_SUMMARY_MAX_CHARS,
    FinalAssessment,
    VerificationOutcome,
    VerificationResult,
    VerificationStatus,
)
from repo_rivet.verification.runtime import VerificationRuntime


class EventSink(Protocol):
    """Minimal logging interface accepted by the controller."""

    def log(self, event_type: str, **data: Any) -> None:
        """Record a structured agent event."""
        ...


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Final outcome returned to the CLI for every terminal path."""

    status: Literal["success", "incomplete", "blocked", "stopped", "error"]
    summary: str
    reason: str | None
    modified_files: tuple[str, ...]
    step_count: int
    tool_call_count: int
    verification_status: VerificationOutcome


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
        runtime = getattr(tool_registry, "verification_runtime", None)
        workspace = getattr(tool_registry, "workspace", None) or Path.cwd().resolve()
        self.verification_runtime = (
            runtime
            if isinstance(runtime, VerificationRuntime)
            else VerificationRuntime(WorkspacePathPolicy(workspace), CommandPolicy())
        )

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
        repaired_interrupted_calls = self._repair_interrupted_history(memory)
        repaired_empty_assistant_messages = memory.repair_invalid_assistant_messages()
        memory.begin_task_scope()
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
            workspace_revision=memory.workspace_revision,
            verification_plan=memory.verification_plan,
            verification_results=dict(memory.verification_results),
            verification_plan_recovery_attempts=memory.verification_plan_recovery_attempts,
            candidate_final_assessment=memory.candidate_final_assessment,
            reflection_required=memory.reflection_required or bool(memory.invalidated_files),
            provider_reasoning_detected=memory.provider_requires_reasoning_content,
            force_thinking_disabled=self._history_requires_non_thinking(memory),
        )
        self.verification_runtime.bind(memory)
        self._log("session_start", task=state.task)
        if repaired_legacy_reflection:
            self._log(
                "legacy_decision_state_repaired",
                reason="trailing decision-validation rejections did not execute tools",
            )
        if repaired_empty_assistant_messages:
            self._log(
                "invalid_history_repaired",
                removed_empty_assistant_messages=repaired_empty_assistant_messages,
            )
        if repaired_interrupted_calls:
            self._log(
                "interrupted_history_repaired",
                calls=repaired_interrupted_calls,
            )
        self._save_memory(memory, state, status=state.status.value)
        try:
            while True:
                reason = self.termination_policy.check(state)
                if reason:
                    if (
                        reason
                        == (
                            "maximum agent steps reached "
                            f"({self.termination_policy.config.max_steps})"
                        )
                        and state.status == AgentStatus.FINALIZING
                        and self.verifier.completion_report(state).complete
                    ):
                        return self._finish(
                            state,
                            memory,
                            status="success",
                            summary=self._verified_completion_summary(state),
                        )
                    return self._finish(state, memory, status="stopped", reason=reason)

                try:
                    response = self._complete_with_context_recovery(
                        state=state,
                        memory=memory,
                        tool_schemas=(
                            [] if state.status == AgentStatus.FINALIZING else self._tool_schemas()
                        ),
                    )
                except ResponseParseError as error:
                    state.record_model_error(str(error))
                    memory.messages.append(
                        Message.from_chat_message(state.messages[-1], step=state.tool_call_count)
                    )
                    self._log("model_response_invalid", step=state.step_count, error=str(error))
                    self._save_memory(memory, state, status=state.status.value)
                    continue
                except AgentContextOverflowError as error:
                    return self._finish(
                        state,
                        memory,
                        status="stopped",
                        reason=str(error),
                    )
                except ModelRequestError as error:
                    memory.last_model_error = error.record
                    self._log("model_error", **error.record.model_dump(mode="json"))
                    attempt_text = (
                        f" after {error.record.attempt} attempts"
                        if error.record.attempt > 1
                        else ""
                    )
                    details = [
                        f"status {error.record.status_code}"
                        if error.record.status_code is not None
                        else None,
                        f"code {error.record.error_code}"
                        if error.record.error_code is not None
                        else None,
                        f"request {error.record.request_id}"
                        if error.record.request_id is not None
                        else None,
                    ]
                    detail_text = ", ".join(item for item in details if item)
                    suffix = f" ({detail_text})" if detail_text else ""
                    return self._finish(
                        state,
                        memory,
                        status="blocked",
                        reason=(
                            f"model request failed{attempt_text}: {error.record.error_type}{suffix}"
                        ),
                    )
                except Exception as error:
                    return self._finish(
                        state,
                        memory,
                        status="error",
                        reason=f"model request failed: {type(error).__name__}",
                    )

                raw_assistant_message = response.as_assistant_message()
                leaked_tool_protocol = (
                    state.status == AgentStatus.FINALIZING
                    and contains_embedded_tool_protocol(response.content)
                )
                attempted_finalization_tool = bool(response.tool_calls) or leaked_tool_protocol
                if state.status == AgentStatus.FINALIZING and attempted_finalization_tool:
                    if leaked_tool_protocol:
                        self._log(
                            "finalization_protocol_text_discarded",
                            step=state.step_count + 1,
                        )
                    response = replace(
                        response,
                        content=(
                            None
                            if response.tool_calls
                            else self._verified_completion_summary(state)
                        ),
                    )
                assistant_recorded = state.record_model_response(response)
                if response.reasoning_content is not None:
                    state.provider_reasoning_detected = True
                    memory.provider_requires_reasoning_content = True
                length_limited = response.finish_reason == "length" and not response.tool_calls
                assistant_message = response.as_assistant_message()
                memory.total_input_tokens += (
                    response.input_tokens
                    if response.input_tokens is not None
                    else self.context_manager.last_request_tokens
                )
                memory.total_output_tokens += (
                    response.output_tokens
                    if response.output_tokens is not None
                    else self.context_manager.count_message(raw_assistant_message)
                )
                if assistant_recorded:
                    memory.append_assistant(
                        assistant_message,
                        step=state.tool_call_count,
                        ephemeral=length_limited,
                    )
                elif not length_limited:
                    memory.messages.append(
                        Message.from_chat_message(
                            state.messages[-1],
                            step=state.tool_call_count,
                        )
                    )
                self._log(
                    "model_response",
                    step=state.step_count,
                    finish_reason=response.finish_reason,
                    content_length=len(response.content or ""),
                    tools=[call.name for call in response.tool_calls],
                )
                if state.status == AgentStatus.FINALIZING:
                    for call in response.tool_calls:
                        self._log(
                            "tool_call",
                            step=state.step_count,
                            tool_call_id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                        )
                        self._record_blocked_action(
                            call,
                            ToolResult(
                                ok=False,
                                output="",
                                error=(
                                    "All required verification checks already passed; tools are "
                                    "disabled during finalization."
                                ),
                                error_code="finalization_tool_disabled",
                                retryable=False,
                            ),
                            state=state,
                            memory=memory,
                        )
                    return self._finish(
                        state,
                        memory,
                        status="success",
                        summary=(
                            response.content.strip()
                            if response.content and response.content.strip()
                            else self._verified_completion_summary(state)
                        ),
                    )
                if length_limited:
                    if (
                        state.provider_reasoning_detected
                        and state.consecutive_length_responses >= 2
                    ):
                        memory.discard_ephemeral_messages()
                        continuation = (
                            "The provider exhausted the output budget in reasoning twice. "
                            "Thinking is disabled for the next recovery request. Respond "
                            "immediately with one concise tool action or a complete concise "
                            "answer; do not repeat analysis or completed work."
                        )
                    else:
                        continuation = (
                            "The provider truncated the previous response because it reached the "
                            "output limit. Continue now with low reasoning effort. Return one "
                            "concise tool action or a complete concise answer; do not repeat "
                            "analysis or completed work."
                        )
                    state.messages.append({"role": "system", "content": continuation})
                    memory.append_ephemeral_system(
                        continuation,
                        step=state.tool_call_count,
                    )
                    self._log(
                        "model_response_continuation",
                        step=state.step_count,
                        consecutive=state.consecutive_length_responses,
                    )
                    self._save_memory(memory, state, status=state.status.value)
                    continue
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
                    terminal_result = self._handle_final_response(
                        response.content.strip(),
                        state=state,
                        memory=memory,
                    )
                    if terminal_result is not None:
                        return terminal_result
        except KeyboardInterrupt:
            state.interrupted = True
            repaired_interrupted_calls = self._repair_interrupted_history(memory)
            if repaired_interrupted_calls:
                self._log(
                    "interrupted_history_repaired",
                    calls=repaired_interrupted_calls,
                )
            return self._finish(state, memory, status="stopped", reason="interrupted by user")

    def _repair_interrupted_history(self, memory: MemoryState) -> list[str]:
        if self.memory_store is not None:
            return self.memory_store.reconcile_interrupted_tool_calls(memory)

        missing, orphan_results = memory.repair_interrupted_tool_history()
        descriptions = [f"{name} ({call_id})" for call_id, name in missing]
        descriptions.extend(f"orphan tool result ({call_id})" for call_id in orphan_results)
        if not descriptions:
            return []
        warning = (
            "The previous run contained an incomplete tool-call group: "
            + ", ".join(descriptions)
            + ". It was closed without retrying any tool. Inspect current state before repeating "
            "a write or command."
        )
        memory.messages.append(Message(role="system", content=warning))
        if warning not in memory.working.unresolved_errors:
            memory.working.unresolved_errors.append(warning)
        if warning not in memory.summary.unresolved_issues:
            memory.summary.unresolved_issues.append(warning)
        return descriptions

    def _process_tool_turn(
        self,
        calls: list[ToolCall],
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        reasoning_calls = [call for call in calls if call.name == "record_decision"]
        registration_calls = [call for call in calls if call.name == "register_verification"]
        action_calls = [
            call for call in calls if call.name not in {"record_decision", "register_verification"}
        ]
        awaiting_plan_recovery = (
            state.verification_plan_recovery_attempts >= 1
            and state.verification_plan is None
            and bool(state.modified_files)
        )
        if awaiting_plan_recovery and not registration_calls:
            error = "The single verification-plan recovery turn only allows register_verification."
            for call in calls:
                self._log(
                    "tool_call",
                    step=state.step_count,
                    tool_call_id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                )
                result = ToolResult(
                    ok=False,
                    output="",
                    error=error,
                    error_code="verification_plan_missing",
                    retryable=False,
                )
                if call.name in {"record_decision", "register_verification"}:
                    self._record_meta_result(call, result, state=state, memory=memory)
                else:
                    self._record_blocked_action(call, result, state=state, memory=memory)
            return self._finish(
                state,
                memory,
                status="incomplete",
                reason="no verification plan was registered in the recovery turn",
                summary=(
                    state.candidate_final_assessment.summary
                    if state.candidate_final_assessment is not None
                    else ""
                ),
            )
        mutating_calls = [call for call in action_calls if self._is_state_changing(call.name)]
        pending_decision = state.pending_decision
        state.pending_decision = None
        if reasoning_calls or action_calls:
            memory.working.pending_actions.clear()
        reasoning_event: ReasoningEvent | None = None
        reasoning_error: str | None = None
        registration_error: str | None = None
        registered_plan_id: str | None = None
        passed_verification_check = False

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

        if len(registration_calls) > 1:
            registration_error = "Only one verification plan may be registered per model turn."
        elif registration_calls:
            try:
                plan = self.verification_runtime.register_plan(registration_calls[0].arguments)
                state.verification_plan = plan
                state.verification_results = dict(memory.verification_results)
                state.status = AgentStatus.RUNNING
                registered_plan_id = plan.plan_id
            except ValueError as error:
                registration_error = str(error)

        decision_for_validation = reasoning_event
        if reasoning_event is None and mutating_calls:
            decision_for_validation = pending_decision

        validation_error = reasoning_error or registration_error
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
            event_type = (
                "assessment"
                if reasoning_event.phase == ReasoningPhase.FINAL_ASSESSMENT
                else "reasoning"
            )
            self._log(event_type, **reasoning_event.model_dump(mode="json"))
            if reasoning_event.phase == ReasoningPhase.FINAL_ASSESSMENT:
                assessment = FinalAssessment(
                    summary=reasoning_event.summary,
                    changes=sorted(state.modified_files),
                    claimed_completed=True,
                    remaining_risks=reasoning_event.open_questions,
                    evidence_refs=reasoning_event.evidence_refs,
                )
                state.candidate_final_assessment = assessment
                memory.candidate_final_assessment = assessment
            self._save_memory(memory, state, status=state.status.value)

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

            if call.name == "register_verification":
                if registered_plan_id is not None and call == registration_calls[0]:
                    result = ToolResult(
                        ok=True,
                        output=f"Registered verification plan {registered_plan_id}.",
                        metadata={"verification_plan_id": registered_plan_id},
                    )
                    self._log(
                        "verification_plan_registered",
                        plan_id=registered_plan_id,
                        check_ids=[check.check_id for check in (state.verification_plan.checks)],
                    )
                else:
                    result = ToolResult(
                        ok=False,
                        output="",
                        error=registration_error or "Duplicate verification plan registration.",
                        error_code="verification_plan_invalid",
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
            verification_result = (
                self._verification_result(result) if call.name == "run_verification" else None
            )
            terminal_result = self._record_action_result(call, result, state=state, memory=memory)
            if terminal_result is not None:
                return terminal_result
            if verification_result is not None:
                passed_verification_check = (
                    passed_verification_check
                    or verification_result.status == VerificationStatus.PASSED
                    and verification_result.workspace_revision == state.workspace_revision
                )

        if (reasoning_event is not None or registration_calls) and not action_calls:
            if reasoning_event is not None and reasoning_event.phase in {
                ReasoningPhase.PLAN,
                ReasoningPhase.REFLECTION,
            }:
                state.reasoning_only_turns += 1
            else:
                state.reasoning_only_turns = 0
            if reasoning_event is not None and reasoning_event.phase == ReasoningPhase.REFLECTION:
                state.reflection_required = False
                memory.reflection_required = False
            elif (
                reasoning_event is not None
                and reasoning_event.phase == ReasoningPhase.DECISION
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
            self._save_memory(memory, state, status=state.status.value)
        elif action_calls:
            state.reasoning_only_turns = 0
        if validation_error is not None:
            reason = self.termination_policy.check(state)
            if reason:
                return self._finish(state, memory, status="stopped", reason=reason)
        if awaiting_plan_recovery and state.verification_plan is None:
            return self._finish(
                state,
                memory,
                status="incomplete",
                reason="the verification plan supplied during recovery was invalid",
                summary=(
                    state.candidate_final_assessment.summary
                    if state.candidate_final_assessment is not None
                    else ""
                ),
            )
        if passed_verification_check:
            report = self.verifier.completion_report(state)
            if report.runnable:
                terminal_result = self._run_verification_batch(
                    report.runnable,
                    state=state,
                    memory=memory,
                )
                if terminal_result is not None:
                    return terminal_result
                report = self.verifier.completion_report(state)
            self._append_system_feedback(
                state,
                memory,
                {
                    "event": (
                        "verification_complete"
                        if report.complete
                        else "required_verification_incomplete"
                    ),
                    "verification_report": report.model_dump(mode="json"),
                    "instruction": (
                        "All required checks passed. Provide the concise final answer now; do "
                        "not register or run the checks again."
                        if report.complete
                        else "Use the deterministic results to repair the implementation before "
                        "claiming completion."
                    ),
                },
            )
            state.status = AgentStatus.FINALIZING if report.complete else AgentStatus.RUNNING
            if report.complete:
                state.pending_decision = None
                memory.working.pending_actions.clear()
            self._save_memory(memory, state, status=state.status.value)
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
        self._save_memory(memory, state, status=state.status.value)

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
        self._save_memory(memory, state, status=state.status.value)

    def _record_action_result(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
        append_to_conversation: bool = True,
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
        if append_to_conversation:
            state.record_tool_result(call, result)
        else:
            state.record_automatic_tool_result(call, result)
        memory.record_tool_result(
            call,
            result,
            step=state.tool_call_count,
            full_output_path=output_ref,
            append_message=append_to_conversation,
        )
        memory.modified_files = set(state.modified_files)
        memory.workspace_revision = state.workspace_revision
        verification_result = self._verification_result(result)
        if verification_result is not None:
            persisted_result = memory.verification_results.get(
                verification_result.check_id,
                verification_result,
            )
            self.verifier.record(state, persisted_result)
            verification_result = persisted_result
        self._log("observation", **observation.model_dump(mode="json"))
        if verification_result is not None:
            self._log("verification_result", **verification_result.model_dump(mode="json"))
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
        self._save_memory(memory, state, status=state.status.value)

        if result.metadata and result.metadata.get("approval_abort"):
            return self._finish(
                state,
                memory,
                status="stopped",
                reason="agent aborted by user during tool approval",
            )
        reason = self.termination_policy.check(state, include_step_limit=False)
        if reason:
            return self._finish(state, memory, status="stopped", reason=reason)
        return None

    @staticmethod
    def _verified_completion_summary(state: SessionState) -> str:
        if state.candidate_final_assessment is not None and not contains_embedded_tool_protocol(
            state.candidate_final_assessment.summary
        ):
            return state.candidate_final_assessment.summary
        modified = ", ".join(sorted(state.modified_files)) or "none"
        return (
            "Completed the requested task. All required verification checks passed. "
            f"Modified files: {modified}."
        )

    @staticmethod
    def _verification_result(result: ToolResult) -> VerificationResult | None:
        metadata = result.metadata or {}
        value = metadata.get("verification_result")
        if not isinstance(value, dict):
            return None
        try:
            return VerificationResult.model_validate(value)
        except ValueError:
            return None

    def _handle_final_response(
        self,
        summary: str,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        assessment_summary = _bounded_assessment_summary(summary)
        existing_assessment = state.candidate_final_assessment
        if existing_assessment is None or existing_assessment.summary != assessment_summary:
            assessment = (
                FinalAssessment(
                    summary=assessment_summary,
                    changes=sorted(state.modified_files),
                    claimed_completed=True,
                )
                if existing_assessment is None
                else existing_assessment.model_copy(update={"summary": assessment_summary})
            )
            state.candidate_final_assessment = assessment
            memory.candidate_final_assessment = assessment
            self._log("assessment", **assessment.model_dump(mode="json"))

        report = self.verifier.completion_report(state)
        if report.complete:
            return self._finish(state, memory, status="success", summary=summary)

        if state.verification_plan is None:
            if state.verification_plan_recovery_attempts >= 1:
                return self._finish(
                    state,
                    memory,
                    status="incomplete",
                    reason="no executable verification plan was registered after one recovery",
                    summary=summary,
                )
            state.verification_plan_recovery_attempts += 1
            state.status = AgentStatus.AWAITING_VERIFICATION_PLAN
            memory.verification_plan_recovery_attempts = state.verification_plan_recovery_attempts
            self._append_system_feedback(
                state,
                memory,
                {
                    "error": "verification_plan_missing",
                    "modified_files": sorted(state.modified_files),
                    "allowed_next_actions": ["register_verification"],
                    "recovery_attempts_remaining": 0,
                },
            )
            self._save_memory(memory, state, status=state.status.value)
            return None

        runnable = report.runnable
        if not runnable:
            return self._finish(
                state,
                memory,
                status="incomplete",
                reason="required verification checks did not pass",
                summary=summary,
            )

        state.status = AgentStatus.VERIFYING
        self._save_memory(memory, state, status=state.status.value)
        terminal_result = self._run_verification_batch(
            runnable,
            state=state,
            memory=memory,
        )
        if terminal_result is not None:
            return terminal_result

        report = self.verifier.completion_report(state)
        if report.complete:
            return self._finish(state, memory, status="success", summary=summary)

        self._append_system_feedback(
            state,
            memory,
            {
                "error": "required_verification_incomplete",
                "verification_report": report.model_dump(mode="json"),
                "instruction": (
                    "Use the deterministic verification results to repair the implementation. "
                    "File changes make prior results stale; do not claim completion until all "
                    "required checks pass."
                ),
            },
        )
        state.status = AgentStatus.RUNNING
        self._save_memory(memory, state, status=state.status.value)
        return None

    def _run_verification_batch(
        self,
        check_ids: list[str],
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        """Run registered required checks in order until one does not pass."""
        state.status = AgentStatus.VERIFYING
        self._save_memory(memory, state, status=state.status.value)
        for check_id in check_ids:
            call = ToolCall(
                id=f"controller-verify-{uuid4().hex[:12]}",
                name="run_verification",
                arguments={"check_id": check_id},
            )
            self._log(
                "action",
                step=state.step_count,
                tool_call_id=call.id,
                tool=call.name,
                argument_summary=check_id,
                controller_scheduled=True,
            )
            result = self.tool_registry.execute(call)
            if self._verification_result(result) is None:
                now = datetime.now(UTC)
                failed = VerificationResult(
                    check_id=check_id,
                    status=VerificationStatus.ERROR,
                    workspace_revision=state.workspace_revision,
                    reasons=[result.error or "verification check could not be executed"],
                    started_at=now,
                    finished_at=now,
                )
                result = ToolResult(
                    ok=result.ok,
                    output=result.output,
                    error=result.error,
                    metadata={
                        **(result.metadata or {}),
                        "verification_result": failed.model_dump(mode="json"),
                    },
                    raw_output=result.raw_output,
                    error_code=result.error_code,
                    retryable=result.retryable,
                )
            terminal_result = self._record_action_result(
                call,
                result,
                state=state,
                memory=memory,
                append_to_conversation=False,
            )
            if terminal_result is not None:
                return terminal_result
            recorded = state.verification_results.get(check_id)
            if (
                recorded is None
                or recorded.status != VerificationStatus.PASSED
                or recorded.workspace_revision != state.workspace_revision
            ):
                break
        return None

    @staticmethod
    def _append_system_feedback(
        state: SessionState,
        memory: MemoryState,
        payload: dict[str, Any],
    ) -> None:
        message = {
            "role": "system",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        }
        state.messages.append(message)
        memory.messages.append(Message.from_chat_message(message, step=state.tool_call_count))

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
        return tool_name in {"edit_file", "run_command", "write_file"}

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
        if call.name == "run_verification":
            check_id = call.arguments.get("check_id")
            if isinstance(check_id, str):
                return check_id
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
                options = self._model_request_options(state)
                response = self.model_client.complete(
                    messages=messages,
                    tools=tool_schemas,
                    options=options,
                )
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
            finally:
                self._log(
                    "model_call_finished",
                    step=state.step_count + 1,
                    context_recovery_attempt=attempt,
                )

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

    @staticmethod
    def _model_request_options(state: SessionState) -> ModelRequestOptions | None:
        if state.provider_reasoning_detected and state.consecutive_length_responses >= 2:
            return ModelRequestOptions(thinking_enabled=False)
        if state.provider_reasoning_detected and state.consecutive_length_responses >= 1:
            return ModelRequestOptions(reasoning_effort="low")
        if state.force_thinking_disabled:
            return ModelRequestOptions(thinking_enabled=False)
        return None

    @staticmethod
    def _history_requires_non_thinking(memory: MemoryState) -> bool:
        if not memory.provider_requires_reasoning_content:
            return False
        return any(
            message.role == "assistant"
            and bool(message.tool_calls)
            and message.reasoning_content is None
            for message in memory.messages
        )

    def _finish(
        self,
        state: SessionState,
        memory: MemoryState,
        *,
        status: Literal["success", "incomplete", "blocked", "stopped", "error"],
        reason: str | None = None,
        summary: str = "",
    ) -> AgentResult:
        result = self._result(state, status=status, reason=reason, summary=summary)
        state.status = {
            "success": AgentStatus.COMPLETED,
            "incomplete": AgentStatus.INCOMPLETE,
            "blocked": AgentStatus.BLOCKED,
            "stopped": AgentStatus.STOPPED,
            "error": AgentStatus.ERROR,
        }[status]
        memory_status = "completed" if status == "success" else status
        memory.last_agent_outcome = status
        memory.discard_provider_state()
        self._save_memory(memory, state, status=memory_status)
        self._log(
            "session_end",
            status=result.status,
            reason=result.reason,
            modified_files=result.modified_files,
            step_count=result.step_count,
            tool_call_count=result.tool_call_count,
            verification_status=result.verification_status.value,
        )
        return result

    def _save_memory(
        self,
        memory: MemoryState,
        state: SessionState,
        *,
        status: str,
    ) -> None:
        memory.workspace_revision = state.workspace_revision
        memory.verification_plan = state.verification_plan
        memory.verification_results = dict(state.verification_results)
        memory.verification_plan_recovery_attempts = state.verification_plan_recovery_attempts
        memory.candidate_final_assessment = state.candidate_final_assessment
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
        status: Literal["success", "incomplete", "blocked", "stopped", "error"],
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
            verification_status=Verifier.outcome(state),
        )


class AgentContextOverflowError(RuntimeError):
    """Bounded context recovery was exhausted without a safe provider request."""


def _bounded_assessment_summary(summary: str) -> str:
    """Keep the audit projection bounded without truncating the user-visible answer."""
    if len(summary) <= FINAL_ASSESSMENT_SUMMARY_MAX_CHARS:
        return summary
    marker = "\n\n[Assessment summary truncated; full response remains in conversation history.]"
    prefix_length = FINAL_ASSESSMENT_SUMMARY_MAX_CHARS - len(marker)
    return f"{summary[:prefix_length].rstrip()}{marker}"
