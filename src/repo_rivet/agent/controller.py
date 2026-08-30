"""Explicit single-agent model/tool/verification loop."""

import json
import time
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
from repo_rivet.llm.protocol import (
    checkpoint_unreplayable_tool_turns,
    contains_embedded_tool_protocol,
)
from repo_rivet.memory.models import MemoryState, Message
from repo_rivet.memory.store import MemoryStore
from repo_rivet.planning.classifier import PlanClassifier, summarize_workspace
from repo_rivet.planning.errors import PlanModeViolation
from repo_rivet.planning.models import (
    PlanOperation,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
    WorkflowMode,
)
from repo_rivet.planning.policy import AutoPlanMode, AutoPlanPolicy
from repo_rivet.planning.runtime import PLANNING_TOOL_NAMES, PlanRuntime
from repo_rivet.reasoning.manager import ReasoningManager
from repo_rivet.reasoning.models import ReasoningEvent, ReasoningPhase
from repo_rivet.reasoning.validator import DecisionValidationError, validate_decision_for_actions
from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.skills.errors import SkillError
from repo_rivet.skills.runtime import SkillRuntime
from repo_rivet.tools.base import DecisionPolicy, ToolCall, ToolResult
from repo_rivet.tools.planning import RequestPlanArguments
from repo_rivet.tools.registry import ToolRegistry
from repo_rivet.verification.models import (
    FINAL_ASSESSMENT_SUMMARY_MAX_CHARS,
    FinalAssessment,
    VerificationOutcome,
    VerificationResult,
    VerificationStatus,
)
from repo_rivet.verification.runtime import VerificationRuntime

_MAX_VERIFICATION_PLAN_RECOVERY_ATTEMPTS = 3
_MAX_VERIFICATION_PLAN_REVISION_ATTEMPTS = 3


class EventSink(Protocol):
    """Minimal logging interface accepted by the controller."""

    def log(self, event_type: str, **data: Any) -> None:
        """Record a structured agent event."""
        ...


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Final outcome returned to the CLI for every terminal path."""

    status: Literal["success", "plan_ready", "incomplete", "blocked", "stopped", "error"]
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
        skill_runtime: SkillRuntime | None = None,
        auto_plan_policy: AutoPlanPolicy | None = None,
        plan_classifier: PlanClassifier | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.context_manager = context_manager or ContextManager()
        self.verifier = verifier or Verifier()
        self.termination_policy = termination_policy or TerminationPolicy()
        self.event_logger = event_logger
        self.memory_store = memory_store
        self.reasoning_manager = reasoning_manager or ReasoningManager()
        self.skill_runtime = skill_runtime
        self.auto_plan_policy = auto_plan_policy or AutoPlanPolicy()
        self.plan_classifier = plan_classifier
        runtime = getattr(tool_registry, "verification_runtime", None)
        workspace = getattr(tool_registry, "workspace", None) or Path.cwd().resolve()
        self.verification_runtime = (
            runtime
            if isinstance(runtime, VerificationRuntime)
            else VerificationRuntime(WorkspacePathPolicy(workspace), CommandPolicy())
        )
        plan_runtime = getattr(tool_registry, "plan_runtime", None)
        self.plan_runtime = (
            plan_runtime
            if isinstance(plan_runtime, PlanRuntime)
            else PlanRuntime(WorkspacePathPolicy(workspace))
        )

    def run(
        self,
        task: str,
        *,
        memory: MemoryState | None = None,
        workflow_mode: WorkflowMode | None = None,
    ) -> AgentResult:
        """Run until verified success, a deterministic stop, or a model API error."""
        if not task.strip():
            raise ValueError("Task must not be empty")

        workspace = getattr(self.tool_registry, "workspace", None) or Path.cwd().resolve()
        memory = memory or MemoryState(session_id=f"memory-{uuid4().hex[:8]}")
        memory.begin_task_scope()
        if workflow_mode is not None:
            memory.workflow_mode = workflow_mode
        auto_plan_eligible = (
            memory.workflow_mode == WorkflowMode.EXECUTE and self._can_start_new_plan(memory)
        )
        auto_plan_reason = (
            self.auto_plan_policy.preflight_reason(task) if auto_plan_eligible else None
        )
        auto_plan_source = "controller"
        if (
            auto_plan_reason is None
            and auto_plan_eligible
            and self.auto_plan_policy.mode == AutoPlanMode.ADAPTIVE
            and self.plan_classifier is not None
        ):
            workspace_summary = summarize_workspace(Path(workspace))
            self._log(
                "auto_plan_review_started",
                workspace_empty=workspace_summary.empty,
                sampled_files=workspace_summary.sampled_files,
                sampled_directories=workspace_summary.sampled_directories,
            )
            review_started = time.monotonic()
            try:
                classification = self.plan_classifier.classify(task, workspace_summary)
            except Exception as error:
                classification = None
                failure = type(error).__name__
            else:
                failure = "invalid_or_unavailable_response" if classification is None else None
            duration = round(time.monotonic() - review_started, 3)
            if classification is None:
                self._log(
                    "auto_plan_review_failed",
                    duration_seconds=duration,
                    reason=failure,
                    fallback="model_may_request_plan",
                )
            else:
                applied = (
                    classification.decision == "plan"
                    and classification.confidence
                    >= self.auto_plan_policy.classifier_confidence_threshold
                )
                self._log(
                    "auto_plan_reviewed",
                    decision=classification.decision,
                    reason=classification.reason,
                    confidence=classification.confidence,
                    confidence_threshold=(self.auto_plan_policy.classifier_confidence_threshold),
                    applied=applied,
                    duration_seconds=duration,
                    input_tokens=classification.input_tokens,
                    output_tokens=classification.output_tokens,
                )
                if applied:
                    auto_plan_reason = classification.reason
                    auto_plan_source = "llm_classifier"
        remaining_plan_steps = 0
        if (
            memory.workflow_mode == WorkflowMode.EXECUTE
            and memory.plan_artifact is not None
            and memory.plan_artifact.status == PlanStatus.EXECUTING
        ):
            remaining_plan_steps = sum(
                step.status != PlanStepStatus.COMPLETED for step in memory.plan_artifact.steps
            )
        step_limit = self.termination_policy.config.max_steps
        repaired_interrupted_calls = self._repair_interrupted_history(memory)
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
            max_steps=step_limit,
        )
        approval_engine = getattr(self.tool_registry, "approval_engine", None)
        if approval_engine is not None:
            approval_engine.sync_memory_rule()
        state = SessionState(
            task=task.strip(),
            status=(
                AgentStatus.PLANNING
                if memory.workflow_mode == WorkflowMode.PLANNING
                else AgentStatus.EXECUTING
                if memory.plan_artifact is not None
                and memory.plan_artifact.status == PlanStatus.EXECUTING
                else AgentStatus.RUNNING
            ),
            workflow_mode=memory.workflow_mode,
            step_limit=step_limit,
            tool_call_count=memory.tool_event_step,
            initial_tool_call_count=memory.tool_event_step,
            modified_files=set(memory.modified_files),
            workspace_revision=memory.workspace_revision,
            verification_plan=memory.verification_plan,
            verification_results=dict(memory.verification_results),
            verification_plan_recovery_attempts=memory.verification_plan_recovery_attempts,
            verification_plan_revision_required=memory.verification_plan_revision_required,
            verification_plan_revision_reason=memory.verification_plan_revision_reason,
            verification_plan_revision_guidance=memory.verification_plan_revision_guidance,
            verification_plan_revision_attempts=memory.verification_plan_revision_attempts,
            candidate_final_assessment=memory.candidate_final_assessment,
            reflection_required=memory.reflection_required or bool(memory.invalidated_files),
            provider_reasoning_detected=memory.provider_requires_reasoning_content,
            sanitize_unreplayable_provider_history=(
                self._history_requires_reasoning_checkpoint(memory)
            ),
        )
        self.verification_runtime.bind(memory)
        self.plan_runtime.bind(memory)
        if self.skill_runtime is not None:
            try:
                self.skill_runtime.restore(memory)
            except SkillError as error:
                return self._finish(
                    state,
                    memory,
                    status="blocked",
                    reason=str(error),
                )
        auto_plan_started = False
        if auto_plan_reason is not None:
            self._enter_planning_mode(state=state, memory=memory)
            auto_plan_started = True
        if (
            state.workflow_mode == WorkflowMode.EXECUTE
            and memory.plan_artifact is not None
            and memory.plan_artifact.status == PlanStatus.EXECUTING
        ):
            stale_reasons = self.plan_runtime.stale_reasons()
            if stale_reasons:
                memory.plan_artifact.status = PlanStatus.STALE
                memory.workflow_mode = WorkflowMode.PLAN_READY
                state.workflow_mode = WorkflowMode.PLAN_READY
                return self._finish(
                    state,
                    memory,
                    status="blocked",
                    reason="approved plan became stale before execution: "
                    + "; ".join(stale_reasons),
                )
        if state.workflow_mode == WorkflowMode.PLANNING:
            self._append_planning_feedback(state=state, memory=memory)
        elif (
            state.workflow_mode == WorkflowMode.EXECUTE
            and memory.plan_artifact is not None
            and memory.plan_artifact.status == PlanStatus.EXECUTING
        ):
            self._append_system_feedback(
                state,
                memory,
                {
                    "workflow_mode": "execute_approved_plan",
                    "plan": memory.plan_artifact.model_dump(mode="json"),
                    "instruction": (
                        "Execute only the current plan step. Read-only inspection is allowed "
                        "for recovery. Use update_plan and return to user review before changing "
                        "scope. Plan approval does not grant tool approval."
                    ),
                },
            )
        self._log(
            "session_start",
            task=state.task,
            progress_checkpoint_window=self.termination_policy.config.max_steps,
            remaining_plan_steps=remaining_plan_steps,
            next_step_checkpoint=step_limit,
            auto_plan_mode=self.auto_plan_policy.mode.value,
            auto_plan_reason=auto_plan_reason if auto_plan_started else None,
        )
        if auto_plan_started:
            self._log(
                "auto_plan_started",
                source=auto_plan_source,
                reason=auto_plan_reason,
            )
        if repaired_interrupted_calls:
            self._log(
                "interrupted_history_repaired",
                calls=repaired_interrupted_calls,
            )
        self._save_memory(memory, state, status=state.status.value)
        try:
            while True:
                terminal_result = self._check_termination(state=state, memory=memory)
                if terminal_result is not None:
                    return terminal_result

                try:
                    tool_schemas = self._tool_schemas(state.workflow_mode)
                    if state.verification_plan_revision_required:
                        tool_schemas = [
                            schema
                            for schema in tool_schemas
                            if schema.get("function", {}).get("name") == "register_verification"
                        ]
                    response = self._complete_with_context_recovery(
                        state=state,
                        memory=memory,
                        tool_schemas=(
                            [] if state.status == AgentStatus.FINALIZING else tool_schemas
                        ),
                    )
                except ResponseParseError as error:
                    bounded_edit_recovery = (
                        error.code == "invalid_tool_arguments_json"
                        and error.tool_name in {"edit_file", "write_file"}
                    )
                    cancelled_pending_decision = False
                    if bounded_edit_recovery:
                        pending_action = (
                            state.pending_decision.next_action
                            if state.pending_decision is not None
                            else None
                        )
                        if (
                            pending_action is not None
                            and pending_action.tool_name == error.tool_name
                        ):
                            state.pending_decision = None
                            cancelled_pending_decision = True
                        memory.working.pending_actions.clear()
                    recovery_instruction = None
                    if bounded_edit_recovery:
                        if error.tool_name == "edit_file":
                            bounded_action = (
                                "Reread the smallest necessary range, then issue one small, "
                                "coherent, snapshot-bound edit_file operation. Continue the "
                                "refactor through separate edits, using each newly returned "
                                "snapshot."
                            )
                        else:
                            bounded_action = (
                                "Create only a minimal valid file with write_file, then read it "
                                "and expand it through separate small snapshot-bound edit_file "
                                "operations."
                            )
                        fresh_decision = (
                            " Record a fresh decision for the smaller action because the "
                            "previous oversized edit decision has been cancelled."
                            if cancelled_pending_decision
                            else " Record the required decision for the smaller action."
                        )
                        recovery_instruction = (
                            "The edit arguments were truncated or malformed. Do not retry the "
                            "whole-file payload and do not attempt to reconstruct the truncated "
                            f"JSON. {bounded_action}{fresh_decision}"
                        )
                    state.record_model_error(
                        str(error),
                        recovery_instruction=recovery_instruction,
                    )
                    memory.messages.append(
                        Message.from_chat_message(state.messages[-1], step=state.tool_call_count)
                    )
                    self._log(
                        "model_response_invalid",
                        step=state.step_count,
                        error=str(error),
                        error_code=error.code,
                        tool_name=error.tool_name,
                        argument_chars=error.argument_chars,
                        recovery=(
                            "bounded_edit" if bounded_edit_recovery else "retry_valid_response"
                        ),
                        pending_decision_cancelled=cancelled_pending_decision,
                    )
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
                if response.reasoning_context_restart_required:
                    # Keep the full local audit history, but do not replay this tool turn
                    # verbatim after thinking is re-enabled: it has no provider reasoning
                    # state. Later requests replace it with a factual protocol checkpoint.
                    state.sanitize_unreplayable_provider_history = True
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
                    provider_thinking_disabled=response.provider_thinking_disabled,
                    reasoning_context_restart_required=(
                        response.reasoning_context_restart_required
                    ),
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
                    replayable_reasoning = bool(
                        response.reasoning_content and response.reasoning_content.strip()
                    )
                    if not replayable_reasoning:
                        # A compatible provider may truncate a thinking response without
                        # exposing the hidden state that it requires on the next request.
                        # Restart from durable facts with thinking disabled instead of sending
                        # an incomplete assistant message that the provider will reject.
                        memory.discard_ephemeral_messages()
                        state.force_thinking_disabled = True
                        continuation = (
                            "The provider truncated the previous response without returning "
                            "replayable reasoning state. Thinking is disabled for this recovery "
                            "request. Restart from the available facts and return one concise "
                            "tool action or a complete concise answer."
                        )
                    elif (
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
                        replayable_reasoning=replayable_reasoning,
                        thinking_disabled=state.force_thinking_disabled,
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
                    if state.workflow_mode == WorkflowMode.PLANNING:
                        self._append_system_feedback(
                            state,
                            memory,
                            {
                                "error": "plan_artifact_required",
                                "instruction": (
                                    "Planning is not complete until submit_plan or update_plan "
                                    "produces a locally validated Plan Artifact. Continue "
                                    "inspection or submit the structured plan now."
                                ),
                            },
                        )
                        self._save_memory(memory, state, status=state.status.value)
                        continue
                    if (
                        memory.plan_artifact is not None
                        and memory.plan_artifact.status == PlanStatus.EXECUTING
                    ):
                        current = memory.plan_artifact.current_step
                        self._append_system_feedback(
                            state,
                            memory,
                            {
                                "error": "plan_execution_incomplete",
                                "current_step": (
                                    current.model_dump(mode="json") if current is not None else None
                                ),
                                "instruction": (
                                    "The approved plan still has incomplete steps. Execute the "
                                    "current step or call update_plan if evidence changed."
                                ),
                            },
                        )
                        self._save_memory(memory, state, status=state.status.value)
                        continue
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

    @staticmethod
    def _can_start_new_plan(memory: MemoryState) -> bool:
        artifact = memory.plan_artifact
        return artifact is None or artifact.status in {PlanStatus.CANCELLED, PlanStatus.COMPLETED}

    @staticmethod
    def _enter_planning_mode(*, state: SessionState, memory: MemoryState) -> None:
        memory.workflow_mode = WorkflowMode.PLANNING
        state.workflow_mode = WorkflowMode.PLANNING
        state.status = AgentStatus.PLANNING
        state.pending_decision = None
        memory.working.pending_actions.clear()

    def _append_planning_feedback(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> None:
        existing_plan = memory.plan_artifact
        plan_tool = (
            "update_plan"
            if existing_plan is not None
            and existing_plan.status not in {PlanStatus.CANCELLED, PlanStatus.COMPLETED}
            else "submit_plan"
        )
        self._append_system_feedback(
            state,
            memory,
            {
                "workflow_mode": "planning",
                "current_plan": (
                    existing_plan.model_dump(mode="json") if existing_plan is not None else None
                ),
                "instruction": (
                    "Inspect the workspace using planning tools only. Submit a structured "
                    f"Plan Artifact with {plan_tool} when it is ready for user review. "
                    "Each create step must name one new file exactly once; write_file creates "
                    "parent directories automatically, so never add a separate directory-creation "
                    "step. "
                    "When updating, retain the exact IDs and specifications of completed "
                    "steps that remain valid; the Controller carries their progress forward. "
                    "Do not reintroduce work already completed outside the remaining plan. "
                    "A tool-free response does not finish planning."
                ),
            },
        )

    def _handle_request_plan(
        self,
        *,
        request_plan_calls: list[ToolCall],
        all_calls: list[ToolCall],
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        selected = request_plan_calls[0]
        error: str | None = None
        reason: str | None = None
        if len(request_plan_calls) > 1:
            error = "Only one request_plan call is allowed per model turn."
        elif state.workflow_mode != WorkflowMode.EXECUTE:
            error = "request_plan is only available in Execute Mode."
        elif not self.auto_plan_policy.model_may_request:
            error = "request_plan is disabled by the configured auto-plan mode."
        elif not self._can_start_new_plan(memory):
            error = "An active plan already exists; revise or finish that plan instead."
        else:
            try:
                arguments = RequestPlanArguments.model_validate(selected.arguments)
                reason = arguments.reason
            except ValueError as validation_error:
                error = f"Invalid request_plan arguments: {validation_error}"

        transitioned = error is None
        mixed_call_error = (
            "request_plan switched the Controller to read-only Plan Mode; co-issued operations "
            "were not executed. Continue with planning tools in the next response."
        )
        for call in all_calls:
            self._log(
                "tool_call",
                step=state.step_count,
                tool_call_id=call.id,
                name=call.name,
                arguments=call.arguments,
            )
            if call == selected and transitioned:
                result = ToolResult(
                    ok=True,
                    output="Controller entered read-only Plan Mode.",
                    metadata={"auto_plan_source": "model", "reason": reason},
                )
                self._record_meta_result(call, result, state=state, memory=memory)
            elif call.name == "request_plan":
                self._record_meta_result(
                    call,
                    ToolResult(
                        ok=False,
                        output="",
                        error=error or "Duplicate request_plan call.",
                        error_code="auto_plan_rejected",
                        retryable=True,
                    ),
                    state=state,
                    memory=memory,
                )
            else:
                self._record_blocked_action(
                    call,
                    ToolResult(
                        ok=False,
                        output="",
                        error=mixed_call_error if transitioned else (error or mixed_call_error),
                        error_code="auto_plan_transition",
                        retryable=True,
                    ),
                    state=state,
                    memory=memory,
                )

        if not transitioned:
            return None
        self._enter_planning_mode(state=state, memory=memory)
        self._append_planning_feedback(state=state, memory=memory)
        self._log("auto_plan_started", source="model", reason=reason)
        self._save_memory(memory, state, status=state.status.value)
        return None

    def _process_tool_turn(
        self,
        calls: list[ToolCall],
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        if state.verification_plan_revision_required and not (
            len(calls) == 1 and calls[0].name == "register_verification"
        ):
            reason = state.verification_plan_revision_reason or "verification was inconclusive"
            error = (
                "Verification-plan revision is required because "
                f"{reason}. The next response must call register_verification only with the "
                "complete replacement plan. record_decision is not required for this meta action."
            )
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
                    error_code="verification_plan_revision_required",
                    retryable=True,
                )
                if call.name in {
                    "record_decision",
                    "register_verification",
                    "submit_plan",
                    "update_plan",
                }:
                    self._record_meta_result(call, result, state=state, memory=memory)
                else:
                    self._record_blocked_action(call, result, state=state, memory=memory)
            return self._retry_verification_plan_revision(state=state, memory=memory)

        calls = [self._normalize_registered_plan_action(call, state=state) for call in calls]

        request_plan_calls = [call for call in calls if call.name == "request_plan"]
        if request_plan_calls:
            return self._handle_request_plan(
                request_plan_calls=request_plan_calls,
                all_calls=calls,
                state=state,
                memory=memory,
            )

        if state.workflow_mode == WorkflowMode.PLANNING:
            forbidden_calls: list[tuple[ToolCall, PlanModeViolation]] = []
            for call in calls:
                try:
                    self.plan_runtime.ensure_tool_allowed(call.name)
                except PlanModeViolation as error:
                    forbidden_calls.append((call, error))
            for call, error in forbidden_calls:
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
                        error=str(error),
                        error_code="plan_mode_violation",
                        retryable=True,
                    ),
                    state=state,
                    memory=memory,
                )
            calls = [call for call in calls if call.name in PLANNING_TOOL_NAMES]
            if not calls:
                return None

        reasoning_calls = [call for call in calls if call.name == "record_decision"]
        registration_calls = [call for call in calls if call.name == "register_verification"]
        plan_calls = [call for call in calls if call.name in {"submit_plan", "update_plan"}]
        action_calls = [
            call
            for call in calls
            if call.name
            not in {"record_decision", "register_verification", "submit_plan", "update_plan"}
        ]
        awaiting_plan_recovery = (
            state.verification_plan_recovery_attempts >= 1
            and state.verification_plan is None
            and bool(state.modified_files)
        )
        if awaiting_plan_recovery and not registration_calls:
            error = (
                "Verification-plan recovery requires register_verification before any other "
                "action. Register the plan now; do not record a decision or repeat the edit yet."
            )
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
                    retryable=True,
                )
                if call.name in {"record_decision", "register_verification"}:
                    self._record_meta_result(call, result, state=state, memory=memory)
                else:
                    self._record_blocked_action(call, result, state=state, memory=memory)
            return self._request_verification_plan_recovery(
                state=state,
                memory=memory,
                trigger="recovery turn did not call register_verification",
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
        plan_artifact_id: str | None = None
        plan_error: str | None = None
        passed_verification_check = False
        inconclusive_verification: VerificationResult | None = None
        successful_action = False

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
            previous_plan = state.verification_plan
            previous_results = dict(memory.verification_results)
            try:
                plan = self.verification_runtime.register_plan(registration_calls[0].arguments)
                if state.verification_plan_revision_required and previous_plan is not None:
                    previous_required_checks = {
                        check.check_id for check in previous_plan.checks if check.required
                    }
                    replacement_required_checks = {
                        check.check_id for check in plan.checks if check.required
                    }
                    missing_requirements = set(previous_plan.requirements) - set(plan.requirements)
                    missing_checks = previous_required_checks - replacement_required_checks
                    if missing_requirements or missing_checks:
                        memory.verification_plan = previous_plan
                        memory.verification_results = previous_results
                        details = [
                            *(f"requirement {item}" for item in sorted(missing_requirements)),
                            *(f"check {item}" for item in sorted(missing_checks)),
                        ]
                        raise ValueError(
                            "Invalid verification plan revision: the complete replacement "
                            "must preserve " + ", ".join(details)
                        )
                state.verification_plan = plan
                state.verification_results = dict(memory.verification_results)
                state.verification_plan_recovery_attempts = 0
                memory.verification_plan_recovery_attempts = 0
                state.verification_plan_revision_required = False
                state.verification_plan_revision_reason = None
                state.verification_plan_revision_guidance = None
                state.verification_plan_revision_attempts = 0
                memory.verification_plan_revision_required = False
                memory.verification_plan_revision_reason = None
                memory.verification_plan_revision_guidance = None
                memory.verification_plan_revision_attempts = 0
                state.status = AgentStatus.RUNNING
                registered_plan_id = plan.plan_id
            except ValueError as error:
                registration_error = str(error)

        if len(plan_calls) > 1:
            plan_error = "Only one plan submission or update is allowed per model turn."
        elif plan_calls:
            plan_call = plan_calls[0]
            if action_calls:
                plan_error = (
                    "submit_plan or update_plan must be the only non-reasoning operation "
                    "in its turn."
                )
            elif plan_call.name == "submit_plan" and state.workflow_mode != WorkflowMode.PLANNING:
                plan_error = "submit_plan is only available in Plan Mode."
            elif plan_call.name == "update_plan" and memory.plan_artifact is None:
                plan_error = "update_plan requires an existing plan artifact."
            else:
                try:
                    artifact = self.plan_runtime.submit(
                        plan_call.arguments,
                        update_reason=(
                            str(plan_call.arguments.get("reason", "")).strip() or None
                            if plan_call.name == "update_plan"
                            else None
                        ),
                    )
                    plan_artifact_id = artifact.plan_id
                except ValueError as error:
                    plan_error = str(error)

        decision_for_validation = reasoning_event
        if reasoning_event is None and mutating_calls:
            decision_for_validation = pending_decision

        requires_decision = any(self._requires_decision(call) for call in mutating_calls)
        if mutating_calls and not requires_decision:
            # A typed registered verification plan is a stronger audit record than a
            # provider-authored decision. An exact run_command may also have been
            # canonicalized to run_verification after the model recorded its intent, so a
            # now-stale advisory tool name must not reject the safer normalized action.
            decision_for_validation = None

        validation_error = reasoning_error or registration_error
        plan_step_validation_error = False
        verification_plan_missing_for_file_change = False
        if plan_calls and action_calls:
            validation_error = plan_error
        if (
            validation_error is None
            and memory.plan_artifact is not None
            and memory.plan_artifact.status == PlanStatus.EXECUTING
        ):
            for call in mutating_calls:
                plan_error = self.plan_runtime.validate_action(call)
                if plan_error is not None:
                    plan_step_validation_error = True
                    validation_error = plan_error
                    break
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
                    require_decision=requires_decision,
                )
            except DecisionValidationError as error:
                validation_error = str(error)
        if (
            validation_error is None
            and state.verification_plan is None
            and any(
                self.tool_registry.modifies_workspace_files(call.name) for call in mutating_calls
            )
        ):
            verification_plan_missing_for_file_change = True
            validation_error = (
                "A verification plan is required before the first workspace file change. "
                "Call register_verification before, or in the same response as, the file tool."
            )
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

            if call.name in {"submit_plan", "update_plan"}:
                if plan_artifact_id is not None and call == plan_calls[0]:
                    artifact = memory.plan_artifact
                    result = ToolResult(
                        ok=True,
                        output=(
                            f"Plan {plan_artifact_id} revision "
                            f"{artifact.artifact_revision if artifact is not None else 1} "
                            "is ready for user review."
                        ),
                        metadata={
                            "plan_id": plan_artifact_id,
                            "artifact_revision": (
                                artifact.artifact_revision if artifact is not None else 1
                            ),
                        },
                    )
                    self._log(
                        "plan_ready",
                        plan_id=plan_artifact_id,
                        artifact_revision=(
                            artifact.artifact_revision if artifact is not None else 1
                        ),
                        workspace_revision=(
                            artifact.workspace_revision if artifact is not None else None
                        ),
                        update_reason=memory.plan_update_reason,
                        artifact=(
                            artifact.model_dump(mode="json") if artifact is not None else None
                        ),
                    )
                else:
                    result = ToolResult(
                        ok=False,
                        output="",
                        error=plan_error or "Invalid plan request.",
                        error_code="plan_invalid",
                        retryable=True,
                    )
                self._record_meta_result(call, result, state=state, memory=memory)
                continue

            if validation_error is not None:
                result = ToolResult(
                    ok=False,
                    output="",
                    error=validation_error,
                    error_code=(
                        "plan_step_violation"
                        if plan_step_validation_error
                        else "verification_plan_missing"
                        if verification_plan_missing_for_file_change
                        else "decision_validation_failed"
                    ),
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
            if self.tool_registry.decision_policy(call.name) == DecisionPolicy.REGISTERED_PLAN:
                self._log(
                    "plan_authorized_action",
                    step=state.step_count,
                    tool_call_id=call.id,
                    tool=call.name,
                    plan_id=(
                        state.verification_plan.plan_id
                        if state.verification_plan is not None
                        else None
                    ),
                    check_id=call.arguments.get("check_id"),
                    requested_by="model",
                )
            if (
                memory.plan_artifact is not None
                and memory.plan_artifact.status == PlanStatus.EXECUTING
                and self._is_state_changing(call.name)
            ):
                plan_step = memory.plan_artifact.current_step
                self.plan_runtime.start_action()
                if plan_step is not None:
                    self._log_plan_step("plan_step_started", memory, plan_step)
            result = self.tool_registry.execute(call)
            successful_action = successful_action or result.ok
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
                if verification_result.status == VerificationStatus.INCONCLUSIVE:
                    inconclusive_verification = verification_result

        if inconclusive_verification is not None:
            self._require_verification_plan_revision(
                inconclusive_verification,
                state=state,
                memory=memory,
            )
        elif state.verification_plan_revision_required and registration_error is not None:
            terminal_result = self._retry_verification_plan_revision(
                state=state,
                memory=memory,
            )
            if terminal_result is not None:
                return terminal_result

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
            if successful_action:
                self._save_memory(memory, state, status=state.status.value)
        if plan_step_validation_error:
            step = memory.plan_artifact.current_step if memory.plan_artifact is not None else None
            allowed_side_effects = self._plan_step_side_effect_tools(step)
            self._append_system_feedback(
                state,
                memory,
                {
                    "error": "plan_step_violation",
                    "current_step": (
                        {
                            "step_id": step.step_id,
                            "title": step.title,
                            "operation": step.operation.value,
                            "target_files": step.target_files,
                            "verification_ids": step.verification_ids,
                        }
                        if step is not None
                        else None
                    ),
                    "allowed_side_effect_tools": sorted(allowed_side_effects),
                    "instruction": (
                        "The rejected action was not executed. Do not repeat it. Use read-only "
                        "inspection if more evidence is needed, then call one allowed tool for "
                        "the current step; call update_plan alone if the approved scope must "
                        "change."
                    ),
                },
            )
            self._save_memory(memory, state, status=state.status.value)
        if successful_action or plan_step_validation_error:
            terminal_result, controller_checks_ran = self._advance_registered_plan_steps(
                state=state,
                memory=memory,
            )
            if terminal_result is not None:
                return terminal_result
            if controller_checks_ran:
                # The approved plan made deterministic progress. A rejected stale tool from
                # the previous step must not count toward the protocol-failure stop condition.
                state.consecutive_protocol_failures = 0
                passed_verification_check = True
        if validation_error is not None:
            terminal_result = self._check_termination(state=state, memory=memory)
            if terminal_result is not None:
                return terminal_result
        if awaiting_plan_recovery and state.verification_plan is None:
            return self._request_verification_plan_recovery(
                state=state,
                memory=memory,
                trigger="the supplied verification plan was invalid",
            )
        if plan_artifact_id is not None:
            state.workflow_mode = WorkflowMode.PLAN_READY
            state.status = AgentStatus.PLAN_READY
            return self._finish(
                state,
                memory,
                status="plan_ready",
                summary="Plan artifact is ready for user review.",
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
            ready_to_finish = report.complete
            self._append_system_feedback(
                state,
                memory,
                {
                    "event": (
                        "verification_complete"
                        if ready_to_finish
                        else "required_verification_incomplete"
                    ),
                    "verification_report": report.model_dump(mode="json"),
                    "instruction": (
                        "All required checks passed. Provide the concise final answer now; do "
                        "not register or run the checks again."
                        if ready_to_finish
                        else "Satisfy the listed deterministic verification requirements before "
                        "claiming completion."
                    ),
                },
            )
            state.status = AgentStatus.FINALIZING if ready_to_finish else AgentStatus.RUNNING
            if ready_to_finish:
                state.pending_decision = None
                memory.working.pending_actions.clear()
            self._save_memory(memory, state, status=state.status.value)
        return None

    def _normalize_registered_plan_action(
        self,
        call: ToolCall,
        *,
        state: SessionState,
    ) -> ToolCall:
        """Canonicalize an exact registered check before policy and plan validation."""
        artifact = self.plan_runtime.memory.plan_artifact if self.plan_runtime.memory else None
        step = artifact.current_step if artifact is not None else None
        if (
            artifact is None
            or artifact.status != PlanStatus.EXECUTING
            or step is None
            or step.operation not in {PlanOperation.COMMAND, PlanOperation.VERIFY}
            or len(step.verification_ids) != 1
            or call.name != "run_command"
        ):
            return call
        command = call.arguments.get("command")
        cwd = call.arguments.get("cwd", ".")
        check_id = step.verification_ids[0]
        if not isinstance(command, str) or not isinstance(cwd, str):
            return call
        try:
            matches = self.verification_runtime.command_matches(
                check_id,
                command=command,
                cwd=cwd,
            )
        except ValueError:
            return call
        if not matches:
            return call
        normalized = ToolCall(
            id=call.id,
            name="run_verification",
            arguments={"check_id": check_id},
        )
        self._log(
            "plan_action_normalized",
            step=state.step_count,
            tool_call_id=call.id,
            requested_tool=call.name,
            normalized_tool=normalized.name,
            check_id=check_id,
            reason="exact_registered_verification_command",
        )
        return normalized

    def _record_meta_result(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> None:
        if call.name in {
            "register_verification",
            "request_plan",
            "submit_plan",
            "update_plan",
        }:
            state.record_meaningful_progress(call, result)
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
        if (
            memory.plan_artifact is not None
            and memory.plan_artifact.status == PlanStatus.EXECUTING
            and self._is_state_changing(call.name)
        ):
            plan_step = memory.plan_artifact.current_step
            self.plan_runtime.observe_action(
                call,
                result,
                evidence_ref=observation.event_id,
            )
            if plan_step is not None:
                self._log_plan_step("plan_step_finished", memory, plan_step)
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
            if verification_result.status == VerificationStatus.INCONCLUSIVE:
                # An inconclusive result changes the next valid action from rerunning the check
                # to replacing its verification plan. Do not let the generic repeated-call guard
                # terminate before the bounded plan-revision recovery can be installed by the
                # caller.
                state.repeated_tool_calls = 0
                state.last_tool_signature = None
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

    def _check_termination(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        reason = self.termination_policy.check(state)
        if reason is None:
            return None
        checkpoint_reason = self.termination_policy.step_checkpoint_reason(state)
        if reason != checkpoint_reason:
            return self._finish(state, memory, status="stopped", reason=reason)
        if (
            state.status == AgentStatus.FINALIZING
            and self.verifier.completion_report(state).complete
        ):
            return self._finish(
                state,
                memory,
                status="success",
                summary=self._verified_completion_summary(state),
            )
        if state.made_progress_since_checkpoint:
            previous_checkpoint = self.termination_policy.step_limit(state)
            state.renew_step_checkpoint(self.termination_policy.config.max_steps)
            self._log(
                "agent_step_checkpoint_renewed",
                previous_checkpoint=previous_checkpoint,
                next_checkpoint=self.termination_policy.step_limit(state),
                progress_revision=state.progress_revision,
            )
            self._save_memory(memory, state, status=state.status.value)
            return None
        return self._finish(
            state,
            memory,
            status="stopped",
            reason=(
                "no meaningful progress during the current agent-step window "
                f"({self.termination_policy.config.max_steps} steps)"
            ),
        )

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
        if state.verification_plan_revision_required:
            return self._retry_verification_plan_revision(
                state=state,
                memory=memory,
                summary=summary,
            )

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
            return self._request_verification_plan_recovery(
                state=state,
                memory=memory,
                trigger="tool-free response received after unplanned file changes",
                summary=summary,
            )

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

    def _request_verification_plan_recovery(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
        trigger: str,
        summary: str | None = None,
    ) -> AgentResult | None:
        """Request a bounded plan-only correction without stopping on the first mistake."""
        if state.verification_plan_recovery_attempts >= (_MAX_VERIFICATION_PLAN_RECOVERY_ATTEMPTS):
            fallback_summary = (
                state.candidate_final_assessment.summary
                if state.candidate_final_assessment is not None
                else ""
            )
            return self._finish(
                state,
                memory,
                status="incomplete",
                reason=(
                    "no executable verification plan was registered after "
                    f"{_MAX_VERIFICATION_PLAN_RECOVERY_ATTEMPTS} recovery attempts"
                ),
                summary=summary if summary is not None else fallback_summary,
            )

        state.verification_plan_recovery_attempts += 1
        memory.verification_plan_recovery_attempts = state.verification_plan_recovery_attempts
        state.status = AgentStatus.AWAITING_VERIFICATION_PLAN
        self._append_system_feedback(
            state,
            memory,
            {
                "error": "verification_plan_missing",
                "trigger": trigger,
                "modified_files": sorted(state.modified_files),
                "allowed_next_actions": ["register_verification"],
                "instruction": (
                    "Your next response must call register_verification only. Do not emit "
                    "progress text, record_decision, or another action in that response."
                ),
                "recovery_attempts_remaining": (
                    _MAX_VERIFICATION_PLAN_RECOVERY_ATTEMPTS
                    - state.verification_plan_recovery_attempts
                ),
            },
        )
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
                authorization_source=DecisionPolicy.REGISTERED_PLAN.value,
                plan_id=(
                    state.verification_plan.plan_id if state.verification_plan is not None else None
                ),
            )
            if (
                memory.plan_artifact is not None
                and memory.plan_artifact.status == PlanStatus.EXECUTING
            ):
                plan_error = self.plan_runtime.validate_action(call)
                if plan_error is not None:
                    self._append_system_feedback(
                        state,
                        memory,
                        {
                            "error": "plan_step_violation",
                            "detail": plan_error,
                            "instruction": (
                                "Follow the current approved plan step or update the plan."
                            ),
                        },
                    )
                    self._save_memory(memory, state, status=state.status.value)
                    break
                plan_step = memory.plan_artifact.current_step
                self.plan_runtime.start_action()
                if plan_step is not None:
                    self._log_plan_step("plan_step_started", memory, plan_step)
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
            if recorded is not None and recorded.status == VerificationStatus.INCONCLUSIVE:
                self._require_verification_plan_revision(
                    recorded,
                    state=state,
                    memory=memory,
                )
                break
            if (
                recorded is None
                or recorded.status != VerificationStatus.PASSED
                or recorded.workspace_revision != state.workspace_revision
            ):
                break
        return None

    def _advance_registered_plan_steps(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> tuple[AgentResult | None, bool]:
        """Controller-schedule typed checks for ready command and verify plan steps."""
        plan = state.verification_plan
        if plan is None:
            return None, False
        registered = {check.check_id for check in plan.checks}
        ran = False
        while True:
            artifact = memory.plan_artifact
            step = artifact.current_step if artifact is not None else None
            if (
                artifact is None
                or artifact.status != PlanStatus.EXECUTING
                or step is None
                or step.status != PlanStepStatus.PENDING
                or step.operation not in {PlanOperation.COMMAND, PlanOperation.VERIFY}
            ):
                break
            check_ids = [check_id for check_id in step.verification_ids if check_id in registered]
            if not check_ids:
                break
            previous_step_id = step.step_id
            self._log(
                "plan_checks_scheduled",
                plan_id=artifact.plan_id,
                step_id=step.step_id,
                operation=step.operation.value,
                check_ids=check_ids,
                reason="current_plan_step_has_registered_checks",
            )
            terminal_result = self._run_verification_batch(
                check_ids,
                state=state,
                memory=memory,
            )
            ran = True
            if terminal_result is not None:
                return terminal_result, ran
            if state.verification_plan_revision_required:
                break
            current = artifact.current_step
            if current is not None and current.step_id == previous_step_id:
                break
        if memory.plan_artifact is not None and memory.plan_artifact.status == PlanStatus.EXECUTING:
            state.status = AgentStatus.EXECUTING
        return None, ran

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

    def _append_verification_revision_feedback(
        self,
        state: SessionState,
        memory: MemoryState,
    ) -> None:
        plan = state.verification_plan
        self._append_system_feedback(
            state,
            memory,
            {
                "error": "verification_plan_revision_required",
                "reason": state.verification_plan_revision_reason,
                "current_plan": plan.model_dump(mode="json") if plan is not None else None,
                "allowed_next_actions": ["register_verification"],
                "instruction": state.verification_plan_revision_guidance
                or (
                    "The next response must call register_verification only with a complete "
                    "replacement plan. Preserve every still-required requirement and check. "
                    "For custom or behavior checks, add a deterministic output or required-"
                    "artifact oracle. register_verification is a Controller meta action and "
                    "does not require record_decision."
                ),
            },
        )

    def _require_verification_plan_revision(
        self,
        result: VerificationResult,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> None:
        reason = "; ".join(result.reasons) or (
            "the check did not provide a deterministic success oracle"
        )
        revision_reason = f"{result.check_id}: {reason}"[:1_000]
        self._set_verification_plan_revision(
            revision_reason,
            guidance=(
                "The next response must call register_verification only with a complete "
                "replacement plan. Preserve every still-required requirement and check. "
                "For custom or behavior checks, add a deterministic output or required-"
                "artifact oracle. register_verification is a Controller meta action and does "
                "not require record_decision."
            ),
            state=state,
            memory=memory,
        )

    def _set_verification_plan_revision(
        self,
        reason: str,
        *,
        guidance: str,
        state: SessionState,
        memory: MemoryState,
    ) -> None:
        if not state.verification_plan_revision_required:
            state.verification_plan_revision_attempts = 0
            memory.verification_plan_revision_attempts = 0
        state.verification_plan_revision_required = True
        state.verification_plan_revision_reason = reason[:1_000]
        state.verification_plan_revision_guidance = guidance[:2_000]
        memory.verification_plan_revision_required = True
        memory.verification_plan_revision_reason = state.verification_plan_revision_reason
        memory.verification_plan_revision_guidance = state.verification_plan_revision_guidance
        state.pending_decision = None
        memory.working.pending_actions.clear()
        state.status = AgentStatus.AWAITING_VERIFICATION_PLAN
        self._append_verification_revision_feedback(state, memory)
        self._save_memory(memory, state, status=state.status.value)

    def _retry_verification_plan_revision(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
        summary: str = "",
    ) -> AgentResult | None:
        state.verification_plan_revision_attempts += 1
        memory.verification_plan_revision_attempts = state.verification_plan_revision_attempts
        if state.verification_plan_revision_attempts >= _MAX_VERIFICATION_PLAN_REVISION_ATTEMPTS:
            return self._finish(
                state,
                memory,
                status="incomplete",
                reason=(
                    "verification plan revision was not provided after "
                    f"{_MAX_VERIFICATION_PLAN_REVISION_ATTEMPTS} recovery attempts: "
                    f"{state.verification_plan_revision_reason or 'revision required'}"
                ),
                summary=summary,
            )
        self._append_verification_revision_feedback(state, memory)
        self._save_memory(memory, state, status=state.status.value)
        return None

    def _tool_schemas(self, workflow_mode: WorkflowMode) -> list[dict[str, Any]]:
        schemas = self.tool_registry.schemas()
        if workflow_mode == WorkflowMode.PLANNING:
            schemas = [
                schema
                for schema in schemas
                if schema.get("function", {}).get("name") in PLANNING_TOOL_NAMES
            ]
        elif workflow_mode == WorkflowMode.EXECUTE:
            has_plan = (
                self.plan_runtime.memory is not None
                and self.plan_runtime.memory.plan_artifact is not None
            )
            schemas = [
                schema
                for schema in schemas
                if schema.get("function", {}).get("name") != "submit_plan"
                and (schema.get("function", {}).get("name") != "update_plan" or has_plan)
                and (
                    schema.get("function", {}).get("name") != "request_plan"
                    or (
                        self.auto_plan_policy.model_may_request
                        and self.plan_runtime.memory is not None
                        and self._can_start_new_plan(self.plan_runtime.memory)
                    )
                )
            ]
            artifact = (
                self.plan_runtime.memory.plan_artifact
                if self.plan_runtime.memory is not None
                else None
            )
            if artifact is not None and artifact.status == PlanStatus.EXECUTING:
                allowed_side_effects = self._plan_step_side_effect_tools(artifact.current_step)
                schemas = [
                    schema
                    for schema in schemas
                    if not self.tool_registry.is_state_changing(
                        str(schema.get("function", {}).get("name", ""))
                    )
                    or schema.get("function", {}).get("name") in allowed_side_effects
                ]
        if self.reasoning_manager.config.enabled:
            return schemas
        return [
            schema
            for schema in schemas
            if schema.get("function", {}).get("name") != "record_decision"
        ]

    @staticmethod
    def _plan_step_side_effect_tools(step: PlanStep | None) -> set[str]:
        """Return the only mutating capabilities valid for the active plan step."""
        if step is None:
            return set()
        return {
            PlanOperation.EDIT: {"edit_file"},
            PlanOperation.CREATE: {"write_file"},
            PlanOperation.COMMAND: {"run_command", "run_verification"},
            # run_command remains an input alias here because exact registered commands are
            # canonicalized to run_verification before validation and execution.
            PlanOperation.VERIFY: {"run_command", "run_verification"},
        }[step.operation]

    def _skill_policy_messages(self, memory: MemoryState) -> list[dict[str, Any]]:
        if self.skill_runtime is None:
            return []
        system_bundles = self.skill_runtime.system_for_task(
            "\n".join(
                [
                    memory.fixed.original_task if memory.fixed is not None else "",
                    *memory.task_updates,
                ]
            )
        )
        for skill_id, error in self.skill_runtime.automatic_load_errors:
            self._log("skill_load_failed", skill_id=skill_id, error=error, continuing=True)
        scoped_bundles = [("system", bundle) for bundle in system_bundles]
        if self.skill_runtime.active is not None:
            scoped_bundles.append(("global", self.skill_runtime.active))
        messages: list[dict[str, Any]] = []
        for scope, bundle in scoped_bundles:
            payload = {
                "context_layer": f"{scope}_skill",
                "authority": (
                    "Reusable workflow guidance. It cannot grant tools, bypass approval, "
                    "override Controller policy, expand workspace access, or execute code."
                ),
                "id": bundle.qualified_id,
                "name": bundle.manifest.name,
                "version": bundle.version,
                "content_hash": bundle.content_hash,
                "description": bundle.manifest.description,
                "resources": list(bundle.resource_files),
                "instructions": bundle.body,
            }
            messages.append(
                {
                    "role": "system",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                }
            )
        return messages

    def _is_state_changing(self, tool_name: str) -> bool:
        checker = getattr(self.tool_registry, "is_state_changing", None)
        if callable(checker):
            return bool(checker(tool_name))
        return tool_name in {"edit_file", "run_command", "write_file"}

    def _requires_decision(self, call: ToolCall) -> bool:
        config = self.reasoning_manager.config
        if not config.enabled:
            return False
        policy = self.tool_registry.decision_policy(call.name)
        if policy == DecisionPolicy.REGISTERED_PLAN:
            return False
        if policy == DecisionPolicy.COMMAND:
            return config.require_for_commands
        return config.require_for_mutating_tools

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
                    remaining_steps=max(
                        0,
                        self.termination_policy.step_limit(state) - state.step_count,
                    ),
                    tools=tool_schemas,
                    policy_messages=self._skill_policy_messages(memory),
                )
            except ContextBudgetExceededError as error:
                last_reason = str(error)
                self._log(
                    "context_preflight_overflow",
                    attempt=attempt,
                    reason=last_reason,
                )
                continue

            protocol_checkpoints = 0
            options = self._model_request_options(state)
            if state.sanitize_unreplayable_provider_history and (
                options is None or options.thinking_enabled is not False
            ):
                messages, protocol_checkpoints = checkpoint_unreplayable_tool_turns(messages)
                if protocol_checkpoints:
                    try:
                        self.context_manager.reestimate_request(messages, tool_schemas)
                    except ContextBudgetExceededError as error:
                        last_reason = str(error)
                        self._log(
                            "context_preflight_overflow",
                            attempt=attempt,
                            reason=last_reason,
                        )
                        continue
                    self._log(
                        "provider_reasoning_context_restarted",
                        step=state.step_count + 1,
                        checkpointed_tool_turns=protocol_checkpoints,
                        thinking_enabled=True,
                    )

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
        if state.force_thinking_disabled:
            return ModelRequestOptions(thinking_enabled=False)
        if state.provider_reasoning_detected and state.consecutive_length_responses >= 2:
            return ModelRequestOptions(thinking_enabled=False)
        if state.provider_reasoning_detected and state.consecutive_length_responses >= 1:
            return ModelRequestOptions(reasoning_effort="low")
        return None

    @staticmethod
    def _history_requires_reasoning_checkpoint(memory: MemoryState) -> bool:
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
        status: Literal["success", "plan_ready", "incomplete", "blocked", "stopped", "error"],
        reason: str | None = None,
        summary: str = "",
    ) -> AgentResult:
        result = self._result(state, status=status, reason=reason, summary=summary)
        state.status = {
            "success": AgentStatus.COMPLETED,
            "plan_ready": AgentStatus.PLAN_READY,
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
        memory.verification_plan_revision_required = state.verification_plan_revision_required
        memory.verification_plan_revision_reason = state.verification_plan_revision_reason
        memory.verification_plan_revision_guidance = state.verification_plan_revision_guidance
        memory.verification_plan_revision_attempts = state.verification_plan_revision_attempts
        memory.candidate_final_assessment = state.candidate_final_assessment
        memory.workflow_mode = state.workflow_mode
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

    def _log_plan_step(self, event_type: str, memory: MemoryState, step: Any) -> None:
        artifact = memory.plan_artifact
        if artifact is None:
            return
        self._log(
            event_type,
            plan_id=artifact.plan_id,
            artifact_revision=artifact.artifact_revision,
            step_id=step.step_id,
            step_index=artifact.steps.index(step) + 1,
            step_count=len(artifact.steps),
            title=step.title,
            operation=step.operation.value,
            status=step.status.value,
            error=step.last_error,
        )

    @staticmethod
    def _result(
        state: SessionState,
        *,
        status: Literal["success", "plan_ready", "incomplete", "blocked", "stopped", "error"],
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
