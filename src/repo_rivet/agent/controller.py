"""Explicit single-agent model/tool/verification loop."""

import copy
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from repo_rivet.actions.identity import ActionIdentity
from repo_rivet.actions.models import (
    ActionRecord,
    ActionResultSnapshot,
    ActionStatus,
    DuplicateDisposition,
    RetryClass,
)
from repo_rivet.actions.registry import ActionRegistry, ProposalClassification
from repo_rivet.actions.retry_policy import may_retry_internally, retry_class_for
from repo_rivet.agent.completion_gate import CompletionGate
from repo_rivet.agent.phases import (
    MODEL_PHASES,
    DecisionEpoch,
    ModelCallRecord,
    RunStatus,
    WorkflowPhase,
)
from repo_rivet.agent.runtime import AgentRuntimeState
from repo_rivet.agent.runtime_kernel import RuntimeKernel
from repo_rivet.agent.state import SessionState
from repo_rivet.agent.termination import TerminationPolicy
from repo_rivet.agent.verifier import Verifier
from repo_rivet.approval.models import ApprovalMode
from repo_rivet.context.manager import (
    SYSTEM_PROMPT,
    ContextBudgetExceededError,
    ContextManager,
)
from repo_rivet.editing.models import (
    RECOVERY_MAX_EDIT_OPERATIONS,
    RECOVERY_MAX_NEW_LINES,
)
from repo_rivet.events.models import DomainEventKind
from repo_rivet.llm.base import (
    ModelClient,
    ModelContextLengthError,
    ModelRequestOptions,
    ModelResponse,
    ModelStreamInterrupted,
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
    PlanArtifact,
    PlanOperation,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
    WorkflowMode,
)
from repo_rivet.planning.policy import AutoPlanMode, AutoPlanPolicy
from repo_rivet.planning.runtime import (
    PLANNING_AUXILIARY_TOOL_NAMES,
    PLANNING_TOOL_NAMES,
    PlanRuntime,
)
from repo_rivet.reasoning.manager import ReasoningManager
from repo_rivet.reasoning.models import ReasoningEvent, ReasoningPhase
from repo_rivet.reasoning.policy import (
    AdaptiveReasoningPolicy,
    ReasoningCallPhase,
    ReasoningContext,
    ReasoningPolicyMode,
    ReasoningPolicySettings,
    ReasoningUsage,
)
from repo_rivet.reasoning.validator import DecisionValidationError, validate_decision_for_actions
from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.skills.errors import SkillError
from repo_rivet.skills.runtime import SkillRuntime
from repo_rivet.tools.base import DecisionPolicy, ToolCall, ToolResult
from repo_rivet.tools.filesystem import MAX_READ_LINES
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
_MAX_PLAN_SCOPE_REVISION_ATTEMPTS = 3
_PLAN_SCOPE_RECOVERY_TOOLS = frozenset(
    {"git_diff", "git_status", "list_files", "read_file", "search_text", "update_plan"}
)
_RECOVERY_MAX_WRITE_CHARS = 2_000


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
        system_prompt: str = SYSTEM_PROMPT,
        safety_rules: tuple[str, ...] | None = None,
        completion_rules: tuple[str, ...] | None = None,
        terminal_tool_names: frozenset[str] = frozenset(),
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.context_manager = context_manager or ContextManager()
        self.verifier = verifier or Verifier()
        self.termination_policy = termination_policy or TerminationPolicy()
        self.event_logger = event_logger
        self.memory_store = memory_store
        self.reasoning_manager = reasoning_manager or ReasoningManager()
        self.reasoning_policy = AdaptiveReasoningPolicy()
        self.action_registry = ActionRegistry()
        self.completion_gate = CompletionGate()
        self._reasoning_policy_mode = self.reasoning_manager.config.effort_policy
        self.skill_runtime = skill_runtime
        self.auto_plan_policy = auto_plan_policy or AutoPlanPolicy()
        self.plan_classifier = plan_classifier
        self.system_prompt = system_prompt
        self.safety_rules = safety_rules or (
            "All file operations must stay inside the configured workspace.",
            "Commands run without a shell and obvious destructive commands are blocked.",
            "Denied tool requests must not be repeated unchanged.",
            "Never expose API keys, tokens, passwords, or local configuration contents.",
        )
        self.completion_rules = completion_rules or (
            "Inspect relevant files before editing.",
            "After file changes, complete a successful verification before finishing.",
            "Report modified files, verification, and unresolved errors explicitly.",
        )
        self.terminal_tool_names = terminal_tool_names
        self._steering_source: Callable[[], list[str]] | None = None
        self._runtime_settings_source: Callable[[], dict[str, str]] | None = None
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

    def set_steering_source(self, source: Callable[[], list[str]] | None) -> None:
        """Attach a thread-safe source of user redirects for the current run."""
        self._steering_source = source

    def set_runtime_settings_source(
        self,
        source: Callable[[], dict[str, str]] | None,
    ) -> None:
        """Attach live settings that are consumed only at controller-safe boundaries."""
        self._runtime_settings_source = source

    def set_reasoning_policy_mode(self, mode: ReasoningPolicyMode | str) -> None:
        """Change adaptive/fixed selection at a controller-safe boundary."""
        self._reasoning_policy_mode = ReasoningPolicyMode(mode)

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
        auto_plan_eligible = (
            memory.workflow_mode == WorkflowMode.EXECUTE and self._can_start_new_plan(memory)
        )
        auto_plan_reason = (
            self.auto_plan_policy.preflight_reason(task) if auto_plan_eligible else None
        )
        # Persist the user task before classifiers, recovery, or runtime setup can emit
        # agent activity. The append-only event sequence is the canonical UI chronology.
        self._log(
            "session_start",
            task=task.strip(),
            progress_checkpoint_window=self.termination_policy.config.max_steps,
            remaining_plan_steps=remaining_plan_steps,
            next_step_checkpoint=step_limit,
            auto_plan_mode=self.auto_plan_policy.mode.value,
            auto_plan_reason=auto_plan_reason,
        )
        auto_plan_source = "controller"
        failure: str | None = None
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
        repaired_interrupted_calls = self._repair_interrupted_history(memory)
        memory.start_task(
            task=task,
            workspace=str(workspace),
            system_prompt=self.system_prompt,
            safety_rules=list(self.safety_rules),
            completion_rules=list(self.completion_rules),
            max_steps=step_limit,
        )
        approval_engine = getattr(self.tool_registry, "approval_engine", None)
        if approval_engine is not None:
            approval_engine.sync_memory_rule()
        initial_phase = (
            WorkflowPhase.PLANNING
            if memory.workflow_mode == WorkflowMode.PLANNING
            else WorkflowPhase.PLAN_REVIEW
            if memory.plan_artifact is not None
            and memory.plan_artifact.status in {PlanStatus.READY, PlanStatus.STALE}
            else WorkflowPhase.DECIDING
        )
        runtime = memory.runtime
        if runtime is None or runtime.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.ERROR,
        }:
            runtime = AgentRuntimeState.create(
                memory.session_id,
                workspace_revision=memory.workspace_revision,
            )
        else:
            runtime = runtime.model_copy(deep=True)
            runtime.revisions.workspace = memory.workspace_revision
        state = SessionState(
            task=task.strip(),
            runtime=runtime,
            workflow_mode=memory.workflow_mode,
            step_limit=step_limit,
            tool_call_count=memory.tool_event_step,
            initial_tool_call_count=memory.tool_event_step,
            modified_files=set(memory.modified_files),
            workspace_revision=memory.workspace_revision,
            verification_plan=memory.verification_plan,
            verification_results=dict(memory.verification_results),
            verification_plan_recovery_attempts=memory.verification_plan_recovery_attempts,
            pending_decision=memory.verification_plan_recovery_decision,
            verification_plan_revision_required=memory.verification_plan_revision_required,
            verification_plan_revision_reason=memory.verification_plan_revision_reason,
            verification_plan_revision_guidance=memory.verification_plan_revision_guidance,
            verification_plan_revision_attempts=memory.verification_plan_revision_attempts,
            plan_scope_revision_required=memory.plan_scope_revision_required,
            plan_scope_revision_reason=memory.plan_scope_revision_reason,
            plan_scope_revision_attempts=memory.plan_scope_revision_attempts,
            candidate_final_assessment=memory.candidate_final_assessment,
            provider_reasoning_detected=memory.provider_requires_reasoning_content,
            sanitize_unreplayable_provider_history=(
                self._history_requires_reasoning_checkpoint(memory)
            ),
        )
        self.verification_runtime.bind(memory)
        self.plan_runtime.bind(memory)
        subagent_manager = getattr(self.tool_registry, "subagent_manager", None)
        if subagent_manager is not None:
            subagent_manager.bind(memory)
        runtime_state = state.runtime
        if runtime_state is None:
            raise RuntimeError("Agent runtime was not initialized")
        if self._runtime_needs_reconciliation(runtime_state):
            active = next(
                (item for item in runtime_state.actions.values() if not item.terminal),
                None,
            )
            recovery = (
                self.action_registry.recovery_for(
                    active,
                    reason_code="interrupted_action_state_unknown",
                )
                if active is not None
                and active.status in {ActionStatus.DISPATCHED, ActionStatus.RUNNING}
                else None
            )
            self._dispatch_runtime_event(
                state,
                DomainEventKind.RUNTIME_RECONCILED,
                payload={
                    "recovery": recovery.model_dump(mode="json") if recovery is not None else None
                },
            )
            reconciled_runtime = state.runtime
            if reconciled_runtime is None:
                raise RuntimeError("Agent runtime disappeared during reconciliation")
            active = (
                reconciled_runtime.actions.get(reconciled_runtime.current_action_id)
                if reconciled_runtime.current_action_id is not None
                else None
            )
            if active is not None and active.status == ActionStatus.OBSERVED:
                reconciled_terminal = self._replay_observed_action(
                    active,
                    state=state,
                    memory=memory,
                )
                if reconciled_terminal is not None:
                    return reconciled_terminal
        self._dispatch_runtime_event(
            state,
            DomainEventKind.RUN_ACTIVATED,
            payload={
                "phase": (
                    WorkflowPhase.RECOVERING.value
                    if state.runtime is not None and state.runtime.recovery is not None
                    else initial_phase.value
                )
            },
        )
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
                    "plan": memory.plan_artifact.as_draft().model_dump(mode="json"),
                    "plan_progress": {
                        step.step_id: step.status.value for step in memory.plan_artifact.steps
                    },
                    "current_step_id": (
                        memory.plan_artifact.current_step.step_id
                        if memory.plan_artifact.current_step is not None
                        else None
                    ),
                    "instruction": (
                        "Execute the current plan step. While it is an edit step, a bounded "
                        "follow-up edit to a previously completed file target is also allowed "
                        "when needed to keep the approved files consistent. Read-only "
                        "inspection is allowed for recovery. Use update_plan and return to user "
                        "review before changing any other scope. Plan approval does not grant "
                        "tool approval."
                    ),
                },
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
        self._save_memory(memory, state, status=self._active_memory_status(state))
        try:
            while True:
                if self._apply_runtime_settings(state=state, memory=memory):
                    self._save_memory(memory, state, status=self._active_memory_status(state))
                if self._apply_pending_steering(state=state, memory=memory):
                    self._save_memory(memory, state, status=self._active_memory_status(state))
                artifact = memory.plan_artifact
                current_plan_step = (
                    artifact.current_step
                    if artifact is not None and artifact.status == PlanStatus.EXECUTING
                    else None
                )
                if (
                    state.verification_plan is None
                    and state.verification_plan_recovery_attempts == 0
                    and current_plan_step is not None
                    and current_plan_step.operation in {PlanOperation.COMMAND, PlanOperation.VERIFY}
                    and current_plan_step.verification_ids
                ):
                    terminal_result = self._request_verification_plan_recovery(
                        state=state,
                        memory=memory,
                        trigger=(
                            "the current approved plan step requires verification checks "
                            "that have not been registered"
                        ),
                        requested_check_ids=list(current_plan_step.verification_ids),
                    )
                    if terminal_result is not None:
                        return terminal_result
                terminal_result = self._check_termination(state=state, memory=memory)
                if terminal_result is not None:
                    return terminal_result

                model_call_id: str | None = None
                decision_epoch_id: str | None = None
                try:
                    tool_schemas = self._tool_schemas(state.workflow_mode, state=state)
                    if state.verification_plan_revision_required or (
                        state.verification_plan_recovery_attempts > 0
                        and state.verification_plan is None
                    ):
                        tool_schemas = [
                            schema
                            for schema in tool_schemas
                            if schema.get("function", {}).get("name") == "register_verification"
                        ]
                    elif state.plan_scope_revision_required:
                        tool_schemas = [
                            schema
                            for schema in tool_schemas
                            if schema.get("function", {}).get("name") in _PLAN_SCOPE_RECOVERY_TOOLS
                        ]
                    model_call_id, decision_epoch_id = self._begin_model_call(
                        state=state,
                        memory=memory,
                    )
                    response = self._complete_with_context_recovery(
                        state=state,
                        memory=memory,
                        tool_schemas=(
                            []
                            if state.runtime is not None
                            and state.runtime.phase == WorkflowPhase.FINALIZING
                            else tool_schemas
                        ),
                    )
                except ModelStreamInterrupted:
                    self._end_model_call(
                        state=state,
                        memory=memory,
                        model_call_id=model_call_id,
                        succeeded=False,
                    )
                    if not self._apply_pending_steering(state=state, memory=memory):
                        return self._finish(
                            state,
                            memory,
                            status="error",
                            reason="model stream was interrupted without a replacement request",
                        )
                    self._log("model_redirected", step=state.step_count + 1)
                    self._save_memory(memory, state, status=self._active_memory_status(state))
                    continue
                except ResponseParseError as error:
                    self._end_model_call(
                        state=state,
                        memory=memory,
                        model_call_id=model_call_id,
                        succeeded=False,
                    )
                    bounded_edit_recovery = (
                        error.code == "invalid_tool_arguments_json"
                        and error.tool_name in {"edit_file", "write_file"}
                    )
                    cancelled_pending_decision = False
                    if bounded_edit_recovery:
                        recovery_tool = cast(
                            Literal["edit_file", "write_file"],
                            error.tool_name,
                        )
                        if state.structured_tool_recovery != recovery_tool:
                            state.structured_tool_recovery_failures = 0
                        state.structured_tool_recovery = recovery_tool
                        state.structured_tool_recovery_failures += 1
                        state.structured_tool_recovery_requires_read = (
                            recovery_tool == "edit_file"
                            and state.structured_tool_recovery_failures == 1
                        )
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
                            memory.verification_plan_recovery_decision = None
                            cancelled_pending_decision = True
                        memory.working.pending_actions.clear()
                    recovery_instruction = None
                    if bounded_edit_recovery:
                        if error.tool_name == "edit_file":
                            bounded_action = (
                                "The Controller has temporarily restricted the next action to "
                                "read_file. Reread only the exact target range. After that, "
                                "issue exactly one snapshot-bound edit_file operation with at "
                                f"most "
                                f"{RECOVERY_MAX_NEW_LINES} new_lines entries. Continue larger "
                                "changes through separate edits using each returned snapshot."
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
                        count_as_empty=not bounded_edit_recovery,
                    )
                    if bounded_edit_recovery:
                        state.record_protocol_failure(str(error))
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
                        recovery_attempt=(
                            state.structured_tool_recovery_failures
                            if bounded_edit_recovery
                            else None
                        ),
                        requires_read=state.structured_tool_recovery_requires_read,
                        pending_decision_cancelled=cancelled_pending_decision,
                    )
                    self._save_memory(memory, state, status=self._active_memory_status(state))
                    continue
                except AgentContextOverflowError as error:
                    self._end_model_call(
                        state=state,
                        memory=memory,
                        model_call_id=model_call_id,
                        succeeded=False,
                    )
                    return self._finish(
                        state,
                        memory,
                        status="stopped",
                        reason=str(error),
                    )
                except ModelRequestError as error:
                    self._end_model_call(
                        state=state,
                        memory=memory,
                        model_call_id=model_call_id,
                        succeeded=False,
                    )
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
                    self._end_model_call(
                        state=state,
                        memory=memory,
                        model_call_id=model_call_id,
                        succeeded=False,
                    )
                    return self._finish(
                        state,
                        memory,
                        status="error",
                        reason=f"model request failed: {type(error).__name__}",
                    )

                redirected = self._apply_pending_steering(state=state, memory=memory)
                if self._apply_runtime_settings(state=state, memory=memory):
                    self._save_memory(memory, state, status=self._active_memory_status(state))
                stale_response = not self._decision_epoch_matches(
                    state,
                    decision_epoch_id,
                )
                self._end_model_call(
                    state=state,
                    memory=memory,
                    model_call_id=model_call_id,
                    succeeded=True,
                )
                if redirected or stale_response:
                    self._log(
                        "model_response_discarded_for_redirect",
                        step=state.step_count + 1,
                        stale_decision_epoch=stale_response,
                    )
                    self._save_memory(memory, state, status=self._active_memory_status(state))
                    continue

                raw_assistant_message = response.as_assistant_message()
                leaked_tool_protocol = (
                    state.runtime is not None
                    and state.runtime.phase == WorkflowPhase.FINALIZING
                    and contains_embedded_tool_protocol(response.content)
                )
                attempted_finalization_tool = bool(response.tool_calls) or leaked_tool_protocol
                if (
                    state.runtime is not None
                    and state.runtime.phase == WorkflowPhase.FINALIZING
                    and attempted_finalization_tool
                ):
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
                if state.runtime is not None and state.runtime.phase == WorkflowPhase.FINALIZING:
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
                    self._save_memory(memory, state, status=self._active_memory_status(state))
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
                    memory.verification_plan_recovery_decision = None
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
                        self._save_memory(memory, state, status=self._active_memory_status(state))
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
                        self._save_memory(memory, state, status=self._active_memory_status(state))
                        continue
                    terminal_result = self._handle_final_response(
                        response.content.strip(),
                        state=state,
                        memory=memory,
                    )
                    if terminal_result is not None:
                        return terminal_result
        except KeyboardInterrupt:
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

        runtime_results = {
            action.tool_call_id: action.result
            for action in (memory.runtime.actions.values() if memory.runtime is not None else ())
            if action.result is not None and not action.result_applied
        }

        def persisted_result(call_id: str, _name: str) -> Message | None:
            result = runtime_results.get(call_id)
            if result is None:
                return None
            return Message.from_chat_message(
                result.to_result().as_tool_message(call_id),
                step=memory.tool_event_step,
            )

        missing, orphan_results = memory.repair_interrupted_tool_history(
            result_for=persisted_result
        )
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

    def _apply_pending_steering(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> bool:
        source = self._steering_source
        instructions = source() if source is not None else []
        normalized = [value.strip() for value in instructions if value.strip()]
        if not normalized:
            return False

        for instruction in normalized:
            memory.append_user_update(instruction)
            self._log(
                "user_steer_applied",
                step=state.step_count + 1,
            )
        if state.runtime is not None:
            state.runtime.revisions.knowledge += 1
        state.task = normalized[-1]
        state.pending_decision = None
        state.verification_plan_recovery_attempts = 0
        memory.verification_plan_recovery_attempts = 0
        memory.verification_plan_recovery_decision = None
        state.candidate_final_assessment = None
        state.consecutive_failures = 0
        state.consecutive_protocol_failures = 0
        state.empty_model_responses = 0
        state.structured_tool_recovery = None
        state.structured_tool_recovery_failures = 0
        state.structured_tool_recovery_requires_read = False
        state.renew_step_checkpoint(self.termination_policy.config.max_steps)
        state.started_at = time.monotonic()
        if state.workflow_mode == WorkflowMode.PLANNING:
            self._change_runtime_phase(state, WorkflowPhase.PLANNING)
        elif (
            memory.plan_artifact is not None and memory.plan_artifact.status == PlanStatus.EXECUTING
        ):
            self._change_runtime_phase(state, WorkflowPhase.DECIDING)
        else:
            self._change_runtime_phase(state, WorkflowPhase.DECIDING)
            state.verification_plan = None
            state.verification_results.clear()
            state.verification_plan_recovery_attempts = 0
            state.verification_plan_revision_required = False
            state.verification_plan_revision_reason = None
            state.verification_plan_revision_guidance = None
            state.verification_plan_revision_attempts = 0
            memory.verification_plan = None
            memory.verification_results.clear()
            memory.verification_plan_recovery_attempts = 0
            memory.verification_plan_revision_required = False
            memory.verification_plan_revision_reason = None
            memory.verification_plan_revision_guidance = None
            memory.verification_plan_revision_attempts = 0
            state.plan_scope_revision_required = False
            state.plan_scope_revision_reason = None
            state.plan_scope_revision_attempts = 0
            memory.plan_scope_revision_required = False
            memory.plan_scope_revision_reason = None
            memory.plan_scope_revision_attempts = 0
        return True

    def _apply_runtime_settings(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> bool:
        source = self._runtime_settings_source
        updates = source() if source is not None else {}
        if not updates:
            return False
        approval_mode = updates.get("approval_mode")
        approval_engine = getattr(self.tool_registry, "approval_engine", None)
        if approval_mode is not None and approval_engine is not None:
            approval_engine.set_mode(ApprovalMode(approval_mode))
            memory.approval_mode_override = ApprovalMode(approval_mode)
            if state.runtime is not None and memory.runtime is not None:
                state.runtime.revisions.approval_policy = memory.runtime.revisions.approval_policy
        if "skill" in updates and self.skill_runtime is not None:
            skill = updates["skill"]
            if skill:
                self.skill_runtime.activate(memory, skill)
            else:
                self.skill_runtime.clear(memory)
            if state.runtime is not None:
                state.runtime.revisions.knowledge += 1
            if memory.plan_artifact is not None and memory.plan_artifact.status in {
                PlanStatus.READY,
                PlanStatus.STALE,
            }:
                self._change_runtime_phase(state, WorkflowPhase.PLAN_REVIEW)
        if (
            updates.get("workflow_mode") == WorkflowMode.PLANNING.value
            and state.workflow_mode != WorkflowMode.PLANNING
        ):
            self._enter_planning_mode(state=state, memory=memory)
            if state.runtime is not None:
                state.runtime.revisions.plan += 1
            self._append_planning_feedback(state=state, memory=memory)
        reasoning_policy = updates.get("reasoning_policy")
        if reasoning_policy is not None:
            self._reasoning_policy_mode = ReasoningPolicyMode(reasoning_policy)
        return True

    @staticmethod
    def _can_start_new_plan(memory: MemoryState) -> bool:
        artifact = memory.plan_artifact
        return artifact is None or artifact.status in {PlanStatus.CANCELLED, PlanStatus.COMPLETED}

    def _enter_planning_mode(self, *, state: SessionState, memory: MemoryState) -> None:
        memory.workflow_mode = WorkflowMode.PLANNING
        state.workflow_mode = WorkflowMode.PLANNING
        self._change_runtime_phase(state, WorkflowPhase.PLANNING)
        state.pending_decision = None
        state.verification_plan_recovery_attempts = 0
        memory.verification_plan_recovery_attempts = 0
        memory.working.pending_actions.clear()
        memory.verification_plan_recovery_decision = None

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
                    existing_plan.as_draft().model_dump(mode="json")
                    if existing_plan is not None
                    else None
                ),
                "completed_step_ids": (
                    [
                        step.step_id
                        for step in existing_plan.steps
                        if step.status == PlanStepStatus.COMPLETED
                    ]
                    if existing_plan is not None
                    else []
                ),
                "instruction": (
                    "Inspect the workspace using planning tools only. Submit a structured "
                    f"Plan Artifact with {plan_tool} when it is ready for user review. "
                    "Each create step must name one new file exactly once; write_file creates "
                    "parent directories automatically, so never add a separate directory-creation "
                    "step. Use a delete step with one exact existing target for delete_path. "
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
        self._save_memory(memory, state, status=self._active_memory_status(state))
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

        plan_scope_recovery_turn = bool(calls) and all(
            call.name in _PLAN_SCOPE_RECOVERY_TOOLS for call in calls
        )
        if state.plan_scope_revision_required and not plan_scope_recovery_turn:
            reason = state.plan_scope_revision_reason or "the action exceeds the approved plan"
            error = (
                f"Plan scope revision is required because {reason}. The next response must "
                "use read-only inspection as needed, then call update_plan with the complete "
                "replacement plan. Do not record a decision or retry the rejected action "
                "before user review."
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
                    error_code="plan_scope_revision_required",
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
            return self._retry_plan_scope_revision(state=state, memory=memory)

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
                except PlanModeViolation as plan_mode_exception:
                    forbidden_calls.append((call, plan_mode_exception))
            for call, plan_violation in forbidden_calls:
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
                        error=str(plan_violation),
                        error_code="plan_mode_violation",
                        retryable=True,
                    ),
                    state=state,
                    memory=memory,
                )
            calls = [
                call
                for call in calls
                if call.name in PLANNING_TOOL_NAMES | PLANNING_AUXILIARY_TOOL_NAMES
            ]
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
            state.verification_plan_recovery_attempts >= 1 and state.verification_plan is None
        )
        if awaiting_plan_recovery and not registration_calls:
            recovery_error_message = (
                "Verification-plan recovery requires register_verification before any other "
                "action. The pending action is held by the Controller; do not repeat it."
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
                    error=recovery_error_message,
                    error_code="verification_plan_missing",
                    retryable=True,
                )
                if call.name in {"record_decision", "register_verification"}:
                    self._record_meta_result(call, result, state=state, memory=memory)
                else:
                    self._record_blocked_action(call, result, state=state, memory=memory)
            state.record_protocol_failure(recovery_error_message)
            self._log(
                "model_response_invalid",
                step=state.step_count,
                error=recovery_error_message,
                recovery="required_tool",
                required_tool="register_verification",
            )
            self._append_system_feedback(
                state,
                memory,
                {
                    "error": "verification_plan_recovery_protocol_violation",
                    "allowed_next_actions": ["register_verification"],
                    "instruction": (
                        "Call register_verification only. The Controller still holds the "
                        "previously authorized action, so do not reproduce it."
                    ),
                },
            )
            self._save_memory(memory, state, status=self._active_memory_status(state))
            return None
        missing_plan_verification_calls = [
            call
            for call in action_calls
            if call.name == "run_verification" and state.verification_plan is None
        ]
        if missing_plan_verification_calls and not registration_calls:
            requested_check_ids = [
                str(call.arguments.get("check_id", "")).strip()
                for call in missing_plan_verification_calls
                if str(call.arguments.get("check_id", "")).strip()
            ]
            missing_plan_error = (
                "No verification plan is registered. The requested check cannot run until "
                "register_verification defines its command and deterministic success criteria."
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
                    error=missing_plan_error,
                    error_code="verification_plan_missing",
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
            return self._request_verification_plan_recovery(
                state=state,
                memory=memory,
                trigger="run_verification requested without a registered verification plan",
                requested_check_ids=requested_check_ids,
            )
        mutating_calls = [call for call in action_calls if self._is_state_changing(call.name)]
        pending_decision = state.pending_decision
        state.pending_decision = None
        if action_calls:
            memory.verification_plan_recovery_decision = None
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
                except ValueError as reasoning_exception:
                    reasoning_error = str(reasoning_exception)

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
                self._change_runtime_phase(state, WorkflowPhase.DECIDING)
                registered_plan_id = plan.plan_id
                if state.runtime is not None and memory.runtime is not None:
                    state.runtime.revisions.verification_plan = (
                        memory.runtime.revisions.verification_plan
                    )
            except ValueError as registration_exception:
                registration_error = str(registration_exception)

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
                    state.plan_scope_revision_required = False
                    state.plan_scope_revision_reason = None
                    state.plan_scope_revision_attempts = 0
                    memory.plan_scope_revision_required = False
                    memory.plan_scope_revision_reason = None
                    memory.plan_scope_revision_attempts = 0
                    if state.runtime is not None and memory.runtime is not None:
                        state.runtime.revisions.plan = memory.runtime.revisions.plan
                except ValueError as plan_exception:
                    plan_error = str(plan_exception)

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
        plan_step_violation_call: ToolCall | None = None
        multiple_primary_actions = False
        multiple_mutating_actions = False
        structured_tool_recovery_violation = False
        verification_plan_missing_for_file_change = False
        unchanged_verification_retry: VerificationResult | None = None
        proposal_classification: ProposalClassification | None = None
        if plan_calls and action_calls:
            validation_error = plan_error
        if validation_error is None and len(mutating_calls) > 1:
            multiple_mutating_actions = True
            validation_error = (
                "At most one state-changing tool may run in a model turn; split the "
                "operations and execute only the current plan step."
            )
        if validation_error is None and len(action_calls) > 1:
            multiple_primary_actions = True
            validation_error = (
                "A model turn may contain at most one primary action. Return one tool action; "
                "record_decision and register_verification may accompany it as metadata."
            )
        if validation_error is None and state.structured_tool_recovery is not None:
            for call in action_calls:
                recovery_error = self._structured_tool_recovery_error(call, state)
                if recovery_error is not None:
                    structured_tool_recovery_violation = True
                    validation_error = recovery_error
                    break
        if (
            validation_error is None
            and memory.plan_artifact is not None
            and memory.plan_artifact.status == PlanStatus.EXECUTING
        ):
            for call in mutating_calls:
                plan_error = self.plan_runtime.validate_action(call)
                if plan_error is not None:
                    plan_step_validation_error = True
                    plan_step_violation_call = call
                    validation_error = plan_error
                    break
        if validation_error is None and len(action_calls) == 1 and state.runtime is not None:
            proposal_classification = self.action_registry.classify(
                action_calls[0],
                runtime=state.runtime,
                context=memory,
                plan_step_id=self._current_plan_step_id(memory),
            )
        if validation_error is None:
            for call in action_calls:
                if proposal_classification is not None and proposal_classification.disposition != (
                    DuplicateDisposition.EXECUTE_NEW
                ):
                    break
                unchanged_verification_retry = self._unchanged_failed_verification(call, state)
                if unchanged_verification_retry is None:
                    continue
                check_id = unchanged_verification_retry.check_id
                validation_error = (
                    f"Verification check {check_id} already "
                    f"{unchanged_verification_retry.status.value} at workspace revision "
                    f"{state.workspace_revision}. Do not rerun it before relevant workspace "
                    "changes. Inspect the existing failure evidence with read-only tools. If "
                    "the approved plan has no repair step for the cause, call update_plan alone "
                    "to add one before editing."
                )
                break
        if validation_error is None and (
            proposal_classification is None
            or proposal_classification.disposition == DuplicateDisposition.EXECUTE_NEW
        ):
            try:
                validate_decision_for_actions(
                    decision_for_validation,
                    action_calls,
                    mutating_calls=mutating_calls,
                    require_decision=requires_decision,
                )
            except DecisionValidationError as decision_exception:
                validation_error = str(decision_exception)
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
        if validation_error is not None and not verification_plan_missing_for_file_change:
            state.record_protocol_failure(validation_error)
        else:
            state.consecutive_protocol_failures = 0

        if verification_plan_missing_for_file_change:
            # The action passed decision validation but cannot be dispatched until its
            # verification prerequisite exists. Preserve the one-shot decision across the
            # Controller-inserted registration turn; the requested action has not run yet.
            state.pending_decision = decision_for_validation
            memory.verification_plan_recovery_decision = decision_for_validation

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
            self._save_memory(memory, state, status=self._active_memory_status(state))

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
                    )
                    if deferred_action:
                        suffix = (
                            " It authorizes that matching state-changing tool only in the "
                            "immediately following model response."
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
                    registered_plan = state.verification_plan
                    if registered_plan is None:
                        raise RuntimeError("Registered verification plan is unavailable")
                    result = ToolResult(
                        ok=True,
                        output=f"Registered verification plan {registered_plan_id}.",
                        metadata={"verification_plan_id": registered_plan_id},
                    )
                    self._log(
                        "verification_plan_registered",
                        plan_id=registered_plan_id,
                        check_ids=[check.check_id for check in registered_plan.checks],
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
                    ready_artifact = memory.plan_artifact
                    ready_artifact_revision = (
                        ready_artifact.artifact_revision if ready_artifact is not None else 1
                    )
                    result = ToolResult(
                        ok=True,
                        output=(
                            f"Plan {plan_artifact_id} revision "
                            f"{ready_artifact_revision} "
                            "is ready for user review."
                        ),
                        metadata={
                            "plan_id": plan_artifact_id,
                            "artifact_revision": ready_artifact_revision,
                        },
                    )
                    self._log(
                        "plan_ready",
                        plan_id=plan_artifact_id,
                        artifact_revision=ready_artifact_revision,
                        workspace_revision=(
                            ready_artifact.workspace_revision
                            if ready_artifact is not None
                            else None
                        ),
                        update_reason=memory.plan_update_reason,
                        artifact=(
                            ready_artifact.model_dump(mode="json")
                            if ready_artifact is not None
                            else None
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
                        else "structured_tool_recovery_violation"
                        if structured_tool_recovery_violation
                        else "action_cardinality_violation"
                        if multiple_primary_actions
                        else "verification_retry_without_change"
                        if unchanged_verification_retry is not None
                        else "verification_plan_missing"
                        if verification_plan_missing_for_file_change
                        else "decision_validation_failed"
                    ),
                    retryable=True,
                )
                self._record_blocked_action(call, result, state=state, memory=memory)
                continue

            if proposal_classification is not None:
                handled, terminal_result = self._handle_action_disposition(
                    call,
                    proposal_classification,
                    state=state,
                    memory=memory,
                )
                if terminal_result is not None:
                    return terminal_result
                if handled:
                    continue

            self._log(
                "action",
                step=state.step_count,
                tool_call_id=call.id,
                tool=call.name,
                argument_summary=self._action_summary(call),
            )
            implementation_plan = self._matching_active_plan(call)
            if implementation_plan is not None:
                artifact, step = implementation_plan
                self._log(
                    "plan_authorized_action",
                    step=state.step_count,
                    tool_call_id=call.id,
                    tool=call.name,
                    plan_id=artifact.plan_id,
                    plan_step_id=step.step_id,
                    requested_by="model",
                    authority="approved_implementation_plan",
                )
            elif self.tool_registry.decision_policy(call.name) == DecisionPolicy.REGISTERED_PLAN:
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
                plan_step = self.plan_runtime.start_action(call)
                if plan_step is not None:
                    self._log_plan_step("plan_step_started", memory, plan_step)
            action = self._propose_action(
                call,
                classification=proposal_classification,
                state=state,
                memory=memory,
            )
            result = self._execute_action_effect(
                call,
                action_id=action.action_id,
                state=state,
                memory=memory,
            )
            successful_action = successful_action or self._action_succeeded(call, result)
            verification_result = (
                self._verification_result(result) if call.name == "run_verification" else None
            )
            terminal_result = self._record_action_result(
                call,
                result,
                state=state,
                memory=memory,
                action_id=action.action_id,
            )
            self._apply_action_observation(
                action.action_id,
                call,
                result,
                state=state,
                memory=memory,
            )
            terminal_result = self._post_action_result(
                result,
                state=state,
                memory=memory,
            )
            if terminal_result is not None:
                return terminal_result
            if call.name in self.terminal_tool_names and result.ok:
                return self._finish(
                    state,
                    memory,
                    status="success",
                    summary=result.output,
                )
            if call.name == "edit_file" and result.error_code == "stale_snapshot":
                recovery_result = self._recover_stale_edit_snapshot(
                    call,
                    result,
                    state=state,
                    memory=memory,
                )
                if recovery_result is not None:
                    return recovery_result
            if verification_result is not None:
                passed_verification_check = (
                    passed_verification_check
                    or verification_result.status == VerificationStatus.PASSED
                    and verification_result.workspace_revision == state.workspace_revision
                )
                if verification_result.status == VerificationStatus.INCONCLUSIVE:
                    inconclusive_verification = verification_result

        if verification_plan_missing_for_file_change:
            return self._request_verification_plan_recovery(
                state=state,
                memory=memory,
                trigger="the first workspace file change requires a verification plan",
            )

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
        elif state.plan_scope_revision_required and plan_error is not None:
            terminal_result = self._retry_plan_scope_revision(
                state=state,
                memory=memory,
                reason=plan_error,
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
            if (
                reasoning_event is not None
                and reasoning_event.phase == ReasoningPhase.DECISION
                and reasoning_event.next_action is not None
            ):
                state.pending_decision = reasoning_event
                if state.verification_plan is None and self.tool_registry.modifies_workspace_files(
                    reasoning_event.next_action.tool_name
                ):
                    memory.verification_plan_recovery_decision = reasoning_event
                    return self._request_verification_plan_recovery(
                        state=state,
                        memory=memory,
                        trigger=("the pending workspace file change requires a verification plan"),
                    )
            elif registration_calls and pending_decision is not None:
                # A Controller-required registration attempt does not consume the already
                # validated authority for the blocked file change, even if the submitted
                # verification plan needs another correction.
                state.pending_decision = pending_decision
                memory.verification_plan_recovery_decision = pending_decision
                if registered_plan_id is not None:
                    self._append_system_feedback(
                        state,
                        memory,
                        {
                            "event": "verification_plan_recovery_completed",
                            "instruction": (
                                "The verification plan is registered. Issue the exact pending "
                                "state-changing action now without another record_decision. "
                                "The preserved decision remains one-shot and is consumed by "
                                "that turn."
                            ),
                        },
                    )
            if state.reasoning_only_turns > self.reasoning_manager.config.max_reflection_only_turns:
                prompt = (
                    "Too many reasoning-only turns. Take a concrete tool action now, provide a "
                    "final answer, or state why progress is impossible."
                )
                state.messages.append({"role": "system", "content": prompt})
                memory.messages.append(
                    Message.from_chat_message(state.messages[-1], step=state.tool_call_count)
                )
            self._save_memory(memory, state, status=self._active_memory_status(state))
        elif action_calls:
            state.reasoning_only_turns = 0
            if successful_action:
                self._save_memory(memory, state, status=self._active_memory_status(state))
        if plan_step_validation_error:
            violation_step = (
                memory.plan_artifact.current_step if memory.plan_artifact is not None else None
            )
            allowed_side_effects = self._plan_step_side_effect_tools(violation_step)
            required_action = self._plan_step_action_contract(violation_step)
            completed_step = self._completed_plan_step_for_call(
                memory.plan_artifact,
                plan_step_violation_call,
            )
            rejected_path = (
                plan_step_violation_call.arguments.get("path")
                if plan_step_violation_call is not None
                else None
            )
            repeated_count = state.recent_errors.count(validation_error or "")
            scope_expansion = False
            if plan_step_violation_call is not None:
                try:
                    scope_expansion = self.plan_runtime.requires_scope_revision(
                        plan_step_violation_call
                    )
                except (OSError, ValueError):
                    scope_expansion = False
            require_plan_update = scope_expansion or repeated_count > 1
            if require_plan_update:
                scope_reason = (
                    f"the rejected target {rejected_path!r} is outside the user-approved "
                    "plan boundary"
                    if scope_expansion
                    else "the current plan contract has rejected the same action repeatedly"
                )
                state.plan_scope_revision_required = True
                state.plan_scope_revision_reason = scope_reason
                state.plan_scope_revision_attempts = 0
                memory.plan_scope_revision_required = True
                memory.plan_scope_revision_reason = scope_reason
                memory.plan_scope_revision_attempts = 0
                state.consecutive_protocol_failures = 0
                self._log(
                    "plan_scope_revision_required",
                    step=state.step_count,
                    reason=scope_reason,
                    current_step_id=(
                        violation_step.step_id if violation_step is not None else None
                    ),
                    rejected_tool=(
                        plan_step_violation_call.name
                        if plan_step_violation_call is not None
                        else None
                    ),
                    rejected_path=rejected_path,
                )
                instruction = (
                    "The rejected action was not executed. The approved plan no longer "
                    "describes the required work. Use read-only tools if current snapshots or "
                    "additional evidence are needed, then call update_plan with a complete "
                    "replacement plan that includes the newly discovered target and preserves "
                    "completed steps and verification requirements. Execution pauses for user "
                    "review after the update; do not record a decision or retry the edit first."
                )
            elif completed_step is not None:
                instruction = (
                    f"The rejected target {rejected_path!r} belongs to already-completed plan "
                    f"step {completed_step.step_id}; do not create, edit, or delete it again. "
                    "The next state-changing call must match required_next_action exactly. "
                    "Read-only inspection is allowed first. If the approved scope must change, "
                    "call update_plan alone."
                )
            else:
                instruction = (
                    "The rejected action was not executed. The next state-changing call must "
                    "match required_next_action exactly. Read-only inspection is allowed first. "
                    "If the approved scope must change, call update_plan alone."
                )
            if repeated_count > 1 and not require_plan_update:
                instruction = (
                    f"This same plan-step violation has now occurred {repeated_count} times. "
                    "Do not repeat or slightly rewrite the rejected request. " + instruction
                )
            self._append_system_feedback(
                state,
                memory,
                {
                    "error": "plan_step_violation",
                    "current_step": (
                        {
                            "step_id": violation_step.step_id,
                            "title": violation_step.title,
                            "operation": violation_step.operation.value,
                            "target_files": violation_step.target_files,
                            "verification_ids": violation_step.verification_ids,
                        }
                        if violation_step is not None
                        else None
                    ),
                    "allowed_side_effect_tools": sorted(allowed_side_effects),
                    "required_next_action": required_action,
                    "plan_scope_revision_required": require_plan_update,
                    "allowed_next_actions": (
                        sorted(_PLAN_SCOPE_RECOVERY_TOOLS) if require_plan_update else None
                    ),
                    "rejected_action": (
                        {
                            "tool": plan_step_violation_call.name,
                            "path": rejected_path,
                            "check_id": plan_step_violation_call.arguments.get("check_id"),
                        }
                        if plan_step_violation_call is not None
                        else None
                    ),
                    "already_completed_step": (
                        {
                            "step_id": completed_step.step_id,
                            "operation": completed_step.operation.value,
                            "target_files": completed_step.target_files,
                        }
                        if completed_step is not None
                        else None
                    ),
                    "instruction": instruction,
                },
            )
            self._save_memory(memory, state, status=self._active_memory_status(state))
        elif multiple_mutating_actions:
            cardinality_step = (
                memory.plan_artifact.current_step if memory.plan_artifact is not None else None
            )
            self._append_system_feedback(
                state,
                memory,
                {
                    "error": "multiple_state_changing_actions",
                    "rejected_tools": [
                        {
                            "tool": call.name,
                            "path": call.arguments.get("path"),
                            "check_id": call.arguments.get("check_id"),
                        }
                        for call in mutating_calls
                    ],
                    "current_step": (
                        {
                            "step_id": cardinality_step.step_id,
                            "title": cardinality_step.title,
                            "operation": cardinality_step.operation.value,
                            "target_files": cardinality_step.target_files,
                            "verification_ids": cardinality_step.verification_ids,
                        }
                        if cardinality_step is not None
                        else None
                    ),
                    "required_next_action": self._plan_step_action_contract(cardinality_step),
                    "instruction": (
                        "None of the state-changing calls in the rejected response executed. "
                        "Retry with exactly one state-changing tool call matching "
                        "required_next_action. Do not batch later plan steps into the same "
                        "model response."
                    ),
                },
            )
            self._save_memory(memory, state, status=self._active_memory_status(state))
        if unchanged_verification_retry is not None:
            self._append_system_feedback(
                state,
                memory,
                {
                    "error": "verification_retry_without_change",
                    "check_id": unchanged_verification_retry.check_id,
                    "verification_status": unchanged_verification_retry.status.value,
                    "workspace_revision": state.workspace_revision,
                    "reasons": unchanged_verification_retry.reasons,
                    "instruction": (
                        "Do not run this check again at the unchanged workspace revision. "
                        "Inspect the recorded failure and relevant files. If a repair falls "
                        "outside the current approved plan step, call update_plan alone to add "
                        "the required repair step."
                    ),
                },
            )
            self._save_memory(memory, state, status=self._active_memory_status(state))
        if successful_action or (
            plan_step_validation_error and not state.plan_scope_revision_required
        ):
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
            state.workflow_mode = WorkflowMode.EXECUTE
            self._change_runtime_phase(state, WorkflowPhase.PLAN_REVIEW)
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
            self._change_runtime_phase(
                state,
                WorkflowPhase.FINALIZING if ready_to_finish else WorkflowPhase.DECIDING,
            )
            if ready_to_finish:
                state.pending_decision = None
                memory.working.pending_actions.clear()
                memory.verification_plan_recovery_decision = None
            self._save_memory(memory, state, status=self._active_memory_status(state))
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
        self._save_memory(memory, state, status=self._active_memory_status(state))

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
        self._save_memory(memory, state, status=self._active_memory_status(state))

    @staticmethod
    def _current_plan_step_id(memory: MemoryState) -> str | None:
        artifact = memory.plan_artifact
        if artifact is None or artifact.status != PlanStatus.EXECUTING:
            return None
        step = artifact.current_step
        return step.step_id if step is not None else None

    @staticmethod
    def _runtime_needs_reconciliation(runtime: AgentRuntimeState) -> bool:
        pending_model = runtime.model_call is not None and (
            runtime.model_call.status.value == "pending"
        )
        active_action = any(not action.terminal for action in runtime.actions.values())
        return pending_model or active_action or runtime.wait is not None

    def _replay_observed_action(
        self,
        action: ActionRecord,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        """Apply a persisted observation after restart without executing its tool again."""
        if action.result is None:
            raise RuntimeError(f"Observed action {action.action_id} has no persisted result")
        call = ToolCall(
            id=action.tool_call_id,
            name=action.tool_name,
            arguments=dict(action.normalized_arguments),
        )
        result = action.result.to_result()
        result_already_linked = any(
            item.role == "tool" and item.tool_call_id == call.id for item in memory.messages
        )
        self._record_action_result(
            call,
            result,
            state=state,
            memory=memory,
            append_to_conversation=(
                not action.result_delivered_to_model and not result_already_linked
            ),
            action_id=action.action_id,
        )
        self._apply_action_observation(
            action.action_id,
            call,
            result,
            state=state,
            memory=memory,
            delivered_to_model=True,
        )
        self._log(
            "action_observation_replayed",
            action_id=action.action_id,
            tool=action.tool_name,
        )
        return self._post_action_result(result, state=state, memory=memory)

    def _propose_action(
        self,
        call: ToolCall,
        *,
        classification: ProposalClassification | None,
        state: SessionState,
        memory: MemoryState,
    ) -> ActionRecord:
        runtime = state.runtime
        if runtime is None:
            raise RuntimeError("The runtime must exist before proposing an action")
        proposal = classification or self.action_registry.classify(
            call,
            runtime=runtime,
            context=memory,
            plan_step_id=self._current_plan_step_id(memory),
        )
        if proposal.disposition != DuplicateDisposition.EXECUTE_NEW:
            raise RuntimeError(
                f"Cannot execute a proposal classified as {proposal.disposition.value}"
            )
        if runtime.recovery is not None:
            self._dispatch_runtime_event(
                state,
                DomainEventKind.RECOVERY_CLEARED,
                payload={"phase": WorkflowPhase.DECIDING.value},
            )
            runtime = state.runtime
            assert runtime is not None
        action = self.action_registry.build_record(
            call,
            semantic_key=proposal.semantic_key,
            revisions=runtime.revisions,
            plan_step_id=self._current_plan_step_id(memory),
        )
        self._dispatch_runtime_event(
            state,
            DomainEventKind.ACTION_PROPOSED,
            correlation_id=action.action_id,
            payload={"action": action.model_dump(mode="json")},
        )
        self._save_memory(memory, state, status=self._active_memory_status(state))
        return action

    def _execute_action_effect(
        self,
        call: ToolCall,
        *,
        action_id: str,
        state: SessionState,
        memory: MemoryState,
    ) -> ToolResult:
        """Execute a proposed action while checkpointing every external boundary."""

        def observe(stage: str, payload: dict[str, Any]) -> None:
            runtime = state.runtime
            if runtime is None:
                raise RuntimeError("The runtime disappeared during action execution")
            action = runtime.actions[action_id]
            event: DomainEventKind | None = None
            if stage == "prepared" and action.status == ActionStatus.PROPOSED:
                event = DomainEventKind.ACTION_PREPARED
            elif stage == "approval_requested":
                event = DomainEventKind.APPROVAL_REQUESTED
            elif stage == "approval_resolved":
                event = DomainEventKind.APPROVAL_RESOLVED
            elif stage == "dispatched":
                event = DomainEventKind.ACTION_DISPATCHED
            elif stage == "running":
                event = DomainEventKind.ACTION_RUNNING
            if event is None:
                return
            self._dispatch_runtime_event(
                state,
                event,
                correlation_id=action_id,
                payload=payload,
            )
            self._save_memory(memory, state, status=self._active_memory_status(state))

        while True:
            result = self.tool_registry.execute_with_lifecycle(call, observe)
            retry_class = retry_class_for(result)
            self._dispatch_runtime_event(
                state,
                DomainEventKind.ACTION_OBSERVED,
                correlation_id=action_id,
                payload={
                    "result": ActionResultSnapshot.from_result(result).model_dump(mode="json"),
                    "retryable": retry_class == RetryClass.TRANSIENT_INFRASTRUCTURE,
                    "retry_class": retry_class.value,
                },
            )
            self._save_memory(memory, state, status=self._active_memory_status(state))
            runtime = state.runtime
            assert runtime is not None
            action = runtime.actions[action_id]
            if not may_retry_internally(result, attempt=action.attempt):
                return result
            self._log(
                "action_retry_scheduled",
                action_id=action_id,
                tool=call.name,
                attempt=action.attempt + 1,
                error_code=result.error_code,
            )
            self._dispatch_runtime_event(
                state,
                DomainEventKind.ACTION_RETRY_SCHEDULED,
                correlation_id=action_id,
                payload={"error_code": result.error_code},
            )
            self._save_memory(memory, state, status=self._active_memory_status(state))

    def _apply_action_observation(
        self,
        action_id: str,
        call: ToolCall,
        result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
        delivered_to_model: bool = True,
    ) -> None:
        runtime = state.runtime
        if runtime is None:
            return
        action = runtime.actions[action_id]
        semantic_key = ActionIdentity.result_key(
            action,
            result,
            context=memory,
            revisions=runtime.revisions,
        )
        action_succeeded = self._action_succeeded(call, result)
        self._dispatch_runtime_event(
            state,
            DomainEventKind.OBSERVATION_APPLIED,
            correlation_id=action_id,
            payload={
                "succeeded": action_succeeded,
                "new_knowledge": True,
                "workspace_revision": state.workspace_revision,
                "semantic_key": semantic_key,
                "next_phase": (
                    WorkflowPhase.DECIDING.value
                    if action_succeeded
                    else WorkflowPhase.RECOVERING.value
                ),
            },
        )
        if delivered_to_model:
            self._dispatch_runtime_event(
                state,
                DomainEventKind.OBSERVATION_DELIVERED,
                correlation_id=action_id,
            )
        if action_succeeded:
            if state.runtime is not None and state.runtime.recovery is not None:
                self._dispatch_runtime_event(
                    state,
                    DomainEventKind.RECOVERY_CLEARED,
                    payload={"phase": WorkflowPhase.DECIDING.value},
                )
        else:
            runtime = state.runtime
            assert runtime is not None
            failed = runtime.actions[action_id]
            recovery = self.action_registry.recovery_for(
                failed,
                reason_code=(
                    result.error_code or "verification_failed"
                    if call.name == "run_verification"
                    else result.error_code or "command_failed"
                    if call.name == "run_command"
                    else result.error_code or "action_failed"
                ),
            )
            self._dispatch_runtime_event(
                state,
                DomainEventKind.RECOVERY_ENTERED,
                correlation_id=action_id,
                payload={"recovery": recovery.model_dump(mode="json")},
            )
            self._log(
                "action_recovery_started",
                action_id=action_id,
                tool=call.name,
                reason_code=recovery.reason_code,
            )
        self._save_memory(memory, state, status=self._active_memory_status(state))

    @staticmethod
    def _action_succeeded(call: ToolCall, result: ToolResult) -> bool:
        if not result.ok:
            return False
        if call.name == "run_command":
            return (result.metadata or {}).get("exit_code") == 0
        if call.name != "run_verification":
            return True
        verification = (result.metadata or {}).get("verification_result")
        return isinstance(verification, dict) and verification.get("status") == (
            VerificationStatus.PASSED.value
        )

    def _handle_action_disposition(
        self,
        call: ToolCall,
        classification: ProposalClassification,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> tuple[bool, AgentResult | None]:
        disposition = classification.disposition
        if disposition == DuplicateDisposition.EXECUTE_NEW:
            return False, None
        previous = classification.previous
        if (
            disposition
            in {
                DuplicateDisposition.REUSE_RESULT,
                DuplicateDisposition.REPLAY_UNDELIVERED_RESULT,
            }
            and previous is not None
            and previous.result is not None
        ):
            result = previous.result.to_result()
            metadata = dict(result.metadata or {})
            metadata.update(
                {
                    "execution_attempted": False,
                    "reused_action_id": previous.action_id,
                    "duplicate_disposition": disposition.value,
                }
            )
            result = ToolResult(
                ok=result.ok,
                output=result.output,
                error=result.error,
                metadata=metadata,
                raw_output=result.raw_output,
                error_code=result.error_code,
                retryable=False,
            )
            self._record_suppressed_action(call, result, state=state, memory=memory)
            if not previous.result_delivered_to_model:
                self._dispatch_runtime_event(
                    state,
                    DomainEventKind.OBSERVATION_DELIVERED,
                    correlation_id=previous.action_id,
                )
            self._log(
                "action_result_reused",
                action_id=previous.action_id,
                tool=call.name,
                disposition=disposition.value,
            )
            self._save_memory(memory, state, status=self._active_memory_status(state))
            return True, None

        reason = {
            DuplicateDisposition.REQUIRE_ALTERNATIVE: (
                "the same action already failed; inspect its evidence and choose a distinct action"
            ),
            DuplicateDisposition.BLOCK: (
                "the recovery state forbids repeating the failed action without new evidence"
            ),
        }.get(disposition, "the action cannot be executed in the current state")
        if call.name == "run_verification" and disposition in {
            DuplicateDisposition.REQUIRE_ALTERNATIVE,
            DuplicateDisposition.BLOCK,
        }:
            reason = (
                "This verification check already failed at the current workspace revision. "
                "Do not rerun it before a relevant workspace change; inspect its evidence "
                "and repair the cause."
            )
        if previous is not None and disposition == DuplicateDisposition.REQUIRE_ALTERNATIVE:
            recovery = self.action_registry.recovery_for(
                previous,
                reason_code="alternative_action_required",
            )
            self._dispatch_runtime_event(
                state,
                DomainEventKind.RECOVERY_ENTERED,
                correlation_id=previous.action_id,
                payload={"recovery": recovery.model_dump(mode="json")},
            )
        result = ToolResult(
            ok=False,
            output="",
            error=reason,
            error_code=(
                "verification_retry_without_change"
                if call.name == "run_verification"
                and disposition
                in {
                    DuplicateDisposition.REQUIRE_ALTERNATIVE,
                    DuplicateDisposition.BLOCK,
                }
                else f"action_{disposition.value}"
            ),
            retryable=False,
            metadata={"execution_attempted": False},
        )
        self._record_blocked_action(call, result, state=state, memory=memory)
        self._append_system_feedback(
            state,
            memory,
            {
                "error": result.error_code,
                "failed_action_id": previous.action_id if previous is not None else None,
                "forbidden_semantic_key": classification.semantic_key,
                "instruction": reason,
            },
        )
        self._save_memory(memory, state, status=self._active_memory_status(state))
        return True, None

    def _record_suppressed_action(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> None:
        """Pair a redundant model call without reporting a tool execution or failure."""
        state.record_blocked_tool_result(call, result)
        memory.append_tool_message(result.as_tool_message(call.id), step=state.tool_call_count)
        self._save_memory(memory, state, status=self._active_memory_status(state))

    def _record_action_result(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
        append_to_conversation: bool = True,
        action_id: str | None = None,
    ) -> AgentResult | None:
        if action_id is not None and action_id in memory.applied_action_ids:
            if append_to_conversation and not any(
                item.role == "tool" and item.tool_call_id == call.id for item in memory.messages
            ):
                message = result.as_tool_message(call.id)
                state.messages.append(message)
                memory.append_tool_message(message, step=state.tool_call_count)
            return None
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
        completed_plan_step: PlanStep | None = None
        next_plan_step: PlanStep | None = None
        if (
            memory.plan_artifact is not None
            and memory.plan_artifact.status == PlanStatus.EXECUTING
            and self._is_state_changing(call.name)
        ):
            plan_step = self.plan_runtime.matching_step(call)
            step_was_completed = (
                plan_step is not None and plan_step.status == PlanStepStatus.COMPLETED
            )
            self.plan_runtime.observe_action(
                call,
                result,
                evidence_ref=observation.event_id,
            )
            if plan_step is not None:
                self._log_plan_step("plan_step_finished", memory, plan_step)
                if not step_was_completed and plan_step.status == PlanStepStatus.COMPLETED:
                    completed_plan_step = plan_step
                    next_plan_step = memory.plan_artifact.current_step
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
        if result.ok and state.structured_tool_recovery is not None:
            if state.structured_tool_recovery_requires_read and call.name == "read_file":
                state.structured_tool_recovery_requires_read = False
                self._log(
                    "structured_tool_recovery_read_completed",
                    tool=state.structured_tool_recovery,
                    path=call.arguments.get("path"),
                    failure_count=state.structured_tool_recovery_failures,
                )
            elif call.name == state.structured_tool_recovery:
                self._log(
                    "structured_tool_recovery_completed",
                    tool=state.structured_tool_recovery,
                    failure_count=state.structured_tool_recovery_failures,
                )
                state.structured_tool_recovery = None
                state.structured_tool_recovery_failures = 0
                state.structured_tool_recovery_requires_read = False
        if completed_plan_step is not None:
            required_action = self._plan_step_action_contract(next_plan_step)
            self._append_system_feedback(
                state,
                memory,
                {
                    "event": "plan_step_completed",
                    "completed_step": {
                        "step_id": completed_plan_step.step_id,
                        "operation": completed_plan_step.operation.value,
                        "target_files": completed_plan_step.target_files,
                        "observation_ref": completed_plan_step.last_observation_ref,
                    },
                    "current_step": (
                        {
                            "step_id": next_plan_step.step_id,
                            "title": next_plan_step.title,
                            "operation": next_plan_step.operation.value,
                            "target_files": next_plan_step.target_files,
                            "verification_ids": next_plan_step.verification_ids,
                        }
                        if next_plan_step is not None
                        else None
                    ),
                    "required_next_action": required_action,
                    "instruction": (
                        "Proceed to the current step and issue at most one state-changing tool "
                        "call per model response. The next state-changing call must match "
                        "required_next_action; while that action is an edit, a bounded corrective "
                        "edit to a previously completed file target is also allowed. Read-only "
                        "inspection is allowed before it."
                        if next_plan_step is not None
                        else "All approved plan steps are complete. Do not issue more "
                        "state-changing tools; provide the final assessment."
                    ),
                },
            )
            self._log(
                "plan_step_transition",
                completed_step_id=completed_plan_step.step_id,
                current_step_id=(next_plan_step.step_id if next_plan_step is not None else None),
                required_next_action=required_action,
            )
        verification_result = self._verification_result(result)
        if verification_result is not None:
            persisted_result = memory.verification_results.get(
                verification_result.check_id,
                verification_result,
            )
            self.verifier.record(state, persisted_result)
            verification_result = persisted_result
            if (
                verification_result.workspace_revision == state.workspace_revision
                and verification_result.status
                in {VerificationStatus.FAILED, VerificationStatus.ERROR}
            ):
                plan_step = (
                    memory.plan_artifact.current_step
                    if memory.plan_artifact is not None
                    and memory.plan_artifact.status == PlanStatus.EXECUTING
                    else None
                )
                self._append_system_feedback(
                    state,
                    memory,
                    {
                        "event": "verification_failed",
                        "check_id": verification_result.check_id,
                        "verification_status": verification_result.status.value,
                        "workspace_revision": state.workspace_revision,
                        "reasons": verification_result.reasons,
                        "current_plan_step": (
                            {
                                "step_id": plan_step.step_id,
                                "operation": plan_step.operation.value,
                                "status": plan_step.status.value,
                            }
                            if plan_step is not None
                            else None
                        ),
                        "instruction": (
                            "Do not rerun this check at the unchanged workspace revision. "
                            "Inspect the saved failure evidence and relevant source with "
                            "read-only tools. Repair the cause before rerunning. If the repair "
                            "is outside the current approved plan step, call update_plan alone "
                            "to add the repair step."
                        ),
                    },
                )
                self._log(
                    "verification_repair_required",
                    check_id=verification_result.check_id,
                    status=verification_result.status.value,
                    workspace_revision=state.workspace_revision,
                    plan_step_id=plan_step.step_id if plan_step is not None else None,
                )
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
        if action_id is not None:
            memory.applied_action_ids.add(action_id)
        self._save_memory(memory, state, status=self._active_memory_status(state))

        return None

    def _post_action_result(
        self,
        result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
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

    def _recover_stale_edit_snapshot(
        self,
        stale_call: ToolCall,
        stale_result: ToolResult,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        """Refresh a stale edit target without letting the rejected request loop."""
        path = stale_call.arguments.get("path")
        if not isinstance(path, str) or not path:
            return None
        read_arguments = self._stale_edit_read_arguments(stale_call, stale_result)
        recovery_call = ToolCall(
            id=f"controller-stale-read-{uuid4().hex[:12]}",
            name="read_file",
            arguments=read_arguments,
        )
        self._log(
            "stale_snapshot_recovery_started",
            step=state.step_count,
            path=path,
            start_line=read_arguments["start_line"],
            end_line=read_arguments["end_line"],
            source_tool_call_id=stale_call.id,
        )
        self._log(
            "action",
            step=state.step_count,
            tool_call_id=recovery_call.id,
            tool=recovery_call.name,
            argument_summary=self._action_summary(recovery_call),
            requested_by="controller",
            reason="refresh stale edit snapshot",
        )
        classification = (
            self.action_registry.classify(
                recovery_call,
                runtime=state.runtime,
                context=memory,
                plan_step_id=self._current_plan_step_id(memory),
            )
            if state.runtime is not None
            else None
        )
        recovery_action = self._propose_action(
            recovery_call,
            classification=classification,
            state=state,
            memory=memory,
        )
        recovery_result = self._execute_action_effect(
            recovery_call,
            action_id=recovery_action.action_id,
            state=state,
            memory=memory,
        )
        terminal_result = self._record_action_result(
            recovery_call,
            recovery_result,
            state=state,
            memory=memory,
            append_to_conversation=False,
            action_id=recovery_action.action_id,
        )
        self._apply_action_observation(
            recovery_action.action_id,
            recovery_call,
            recovery_result,
            state=state,
            memory=memory,
            delivered_to_model=True,
        )
        terminal_result = self._post_action_result(
            recovery_result,
            state=state,
            memory=memory,
        )
        metadata = recovery_result.metadata or {}
        if recovery_result.ok:
            state.consecutive_failures = 0
            feedback = {
                "event": "stale_snapshot_recovered",
                "path": path,
                "fresh_snapshot_id": metadata.get("snapshot_id"),
                "fresh_snapshot_tag": metadata.get("snapshot_tag"),
                "visible_start_line": metadata.get("start_line"),
                "visible_end_line": metadata.get("fully_visible_end_line"),
                "fresh_file_content": recovery_result.output,
                "instruction": (
                    "The rejected edit did not execute. Discard its old snapshot and line "
                    "assumptions. Recompute a small coherent edit from this fresh content, "
                    "using fresh_snapshot_id. Record a fresh matching decision with the new "
                    "edit; do not repeat the stale request verbatim. Read another range first "
                    "if any intended target line is not visible above."
                ),
            }
        else:
            feedback = {
                "event": "stale_snapshot_recovery_failed",
                "path": path,
                "error": recovery_result.error,
                "instruction": (
                    "The rejected edit did not execute and the automatic refresh failed. "
                    "Call read_file for this path before attempting any further edit."
                ),
            }
        self._append_system_feedback(state, memory, feedback)
        self._log(
            "stale_snapshot_recovery_finished",
            step=state.step_count,
            path=path,
            ok=recovery_result.ok,
            start_line=metadata.get("start_line"),
            end_line=metadata.get("fully_visible_end_line"),
        )
        self._save_memory(memory, state, status=self._active_memory_status(state))
        return terminal_result

    @staticmethod
    def _stale_edit_read_arguments(
        call: ToolCall,
        result: ToolResult,
    ) -> dict[str, Any]:
        """Choose one bounded range around the earliest stale edit operation."""
        metadata = result.metadata or {}
        total_lines_value = metadata.get("current_total_lines")
        total_lines = total_lines_value if isinstance(total_lines_value, int) else None
        target_lines: list[int] = []
        for operation in call.arguments.get("operations", []):
            if not isinstance(operation, dict):
                continue
            for key in ("start_line", "end_line", "line"):
                value = operation.get(key)
                if isinstance(value, int) and value >= 1:
                    target_lines.append(value)
            if operation.get("op") == "insert_start":
                target_lines.append(1)
            elif operation.get("op") == "insert_end" and total_lines is not None:
                target_lines.append(max(1, total_lines))
        anchor = min(target_lines, default=1)
        if total_lines is not None:
            anchor = min(anchor, max(1, total_lines))
        start_line = max(1, anchor - 20)
        end_line = start_line + MAX_READ_LINES - 1
        if total_lines is not None:
            end_line = min(end_line, max(1, total_lines))
        return {
            "path": call.arguments.get("path"),
            "start_line": start_line,
            "end_line": end_line,
        }

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
            state.runtime is not None
            and state.runtime.phase == WorkflowPhase.FINALIZING
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
            self._save_memory(memory, state, status=self._active_memory_status(state))
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

    @staticmethod
    def _unchanged_failed_verification(
        call: ToolCall,
        state: SessionState,
    ) -> VerificationResult | None:
        if call.name != "run_verification":
            return None
        check_id = call.arguments.get("check_id")
        if not isinstance(check_id, str):
            return None
        result = state.verification_results.get(check_id)
        if (
            result is None
            or result.workspace_revision != state.workspace_revision
            or result.status not in {VerificationStatus.FAILED, VerificationStatus.ERROR}
        ):
            return None
        return result

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
        if state.plan_scope_revision_required:
            return self._retry_plan_scope_revision(
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
        gate = (
            self.completion_gate.evaluate(
                state.runtime,
                memory,
                verification_complete=report.complete,
            )
            if state.runtime is not None
            else None
        )
        self._log(
            "completion_gate_evaluated",
            verifier_complete=report.complete,
            **(gate.model_dump(mode="json") if gate is not None else {"ready": report.complete}),
        )
        if report.complete and (gate is None or gate.ready):
            return self._finish(state, memory, status="success", summary=summary)
        if report.complete and gate is not None and not gate.ready:
            self._append_system_feedback(
                state,
                memory,
                {
                    "error": "completion_gate_not_satisfied",
                    "completion_gate": gate.model_dump(mode="json"),
                    "instruction": (
                        "The completion claim was not accepted. Continue the current plan, "
                        "resolve the active action or approval, and make verification current "
                        "before finalizing."
                    ),
                },
            )
            self._change_runtime_phase(state, WorkflowPhase.DECIDING)
            self._save_memory(memory, state, status=self._active_memory_status(state))
            return None

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

        self._change_runtime_phase(state, WorkflowPhase.VERIFYING)
        self._save_memory(memory, state, status=self._active_memory_status(state))
        terminal_result = self._run_verification_batch(
            runnable,
            state=state,
            memory=memory,
        )
        if terminal_result is not None:
            return terminal_result

        report = self.verifier.completion_report(state)
        gate = (
            self.completion_gate.evaluate(
                state.runtime,
                memory,
                verification_complete=report.complete,
            )
            if state.runtime is not None
            else None
        )
        self._log(
            "completion_gate_evaluated",
            verifier_complete=report.complete,
            **(gate.model_dump(mode="json") if gate is not None else {"ready": report.complete}),
        )
        if report.complete and (gate is None or gate.ready):
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
        self._change_runtime_phase(state, WorkflowPhase.DECIDING)
        self._save_memory(memory, state, status=self._active_memory_status(state))
        return None

    def _request_verification_plan_recovery(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
        trigger: str,
        summary: str | None = None,
        requested_check_ids: list[str] | None = None,
    ) -> AgentResult | None:
        """Request a bounded plan-only correction without stopping on the first mistake."""
        if state.verification_plan_recovery_attempts >= (_MAX_VERIFICATION_PLAN_RECOVERY_ATTEMPTS):
            state.pending_decision = None
            memory.verification_plan_recovery_decision = None
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
        self._change_runtime_phase(state, WorkflowPhase.RECOVERING)
        state.consecutive_protocol_failures = 0
        self._append_system_feedback(
            state,
            memory,
            {
                "error": "verification_plan_missing",
                "trigger": trigger,
                "modified_files": sorted(state.modified_files),
                "requested_check_ids": requested_check_ids or [],
                "allowed_next_actions": ["register_verification"],
                "pending_action_held": (
                    state.pending_decision.next_action.tool_name
                    if state.pending_decision is not None
                    and state.pending_decision.next_action is not None
                    else None
                ),
                "instruction": (
                    "Your next response must call register_verification only. Do not emit "
                    "progress text, record_decision, or another action in that response. "
                    "The Controller is holding any previously authorized action; do not "
                    "reproduce it until registration succeeds."
                ),
                "recovery_attempts_remaining": (
                    _MAX_VERIFICATION_PLAN_RECOVERY_ATTEMPTS
                    - state.verification_plan_recovery_attempts
                ),
            },
        )
        self._log(
            "verification_plan_recovery_started",
            trigger=trigger,
            requested_check_ids=requested_check_ids or [],
            attempt=state.verification_plan_recovery_attempts,
        )
        self._save_memory(memory, state, status=self._active_memory_status(state))
        return None

    def _run_verification_batch(
        self,
        check_ids: list[str],
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> AgentResult | None:
        """Run registered required checks in order until one does not pass."""
        self._change_runtime_phase(state, WorkflowPhase.VERIFYING)
        self._save_memory(memory, state, status=self._active_memory_status(state))
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
                    self._save_memory(memory, state, status=self._active_memory_status(state))
                    break
                plan_step = memory.plan_artifact.current_step
                self.plan_runtime.start_action()
                if plan_step is not None:
                    self._log_plan_step("plan_step_started", memory, plan_step)
            classification = None
            if state.runtime is not None:
                classification = self.action_registry.classify(
                    call,
                    runtime=state.runtime,
                    context=memory,
                    plan_step_id=self._current_plan_step_id(memory),
                )
            action: ActionRecord | None = None
            if (
                classification is not None
                and classification.disposition == DuplicateDisposition.REUSE_RESULT
                and classification.previous is not None
                and classification.previous.result is not None
            ):
                result = classification.previous.result.to_result()
                self._log(
                    "action_result_reused",
                    action_id=classification.previous.action_id,
                    tool=call.name,
                    disposition=classification.disposition.value,
                    controller_scheduled=True,
                )
            elif classification is None or classification.disposition == (
                DuplicateDisposition.EXECUTE_NEW
            ):
                action = self._propose_action(
                    call,
                    classification=classification,
                    state=state,
                    memory=memory,
                )
                result = self._execute_action_effect(
                    call,
                    action_id=action.action_id,
                    state=state,
                    memory=memory,
                )
            elif classification.previous is not None and classification.previous.result is not None:
                result = classification.previous.result.to_result()
            else:
                result = ToolResult(
                    ok=False,
                    output="",
                    error=(
                        "The verification action is already pending or requires a distinct "
                        "recovery action."
                    ),
                    error_code=f"action_{classification.disposition.value}",
                    retryable=False,
                )
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
                action_id=action.action_id if action is not None else None,
            )
            if action is not None:
                self._apply_action_observation(
                    action.action_id,
                    call,
                    result,
                    state=state,
                    memory=memory,
                    delivered_to_model=True,
                )
                terminal_result = self._post_action_result(
                    result,
                    state=state,
                    memory=memory,
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
            self._change_runtime_phase(state, WorkflowPhase.DECIDING)
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
        memory.verification_plan_recovery_decision = None
        self._change_runtime_phase(state, WorkflowPhase.RECOVERING)
        self._append_verification_revision_feedback(state, memory)
        self._save_memory(memory, state, status=self._active_memory_status(state))

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
        self._save_memory(memory, state, status=self._active_memory_status(state))
        return None

    def _retry_plan_scope_revision(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
        reason: str | None = None,
        summary: str = "",
    ) -> AgentResult | None:
        """Retry one constrained update_plan turn without entering a rejection loop."""
        if reason:
            state.plan_scope_revision_reason = reason[:1_000]
            memory.plan_scope_revision_reason = state.plan_scope_revision_reason
        state.plan_scope_revision_attempts += 1
        memory.plan_scope_revision_attempts = state.plan_scope_revision_attempts
        if state.plan_scope_revision_attempts >= _MAX_PLAN_SCOPE_REVISION_ATTEMPTS:
            return self._finish(
                state,
                memory,
                status="incomplete",
                reason=(
                    "plan scope revision was not provided after "
                    f"{_MAX_PLAN_SCOPE_REVISION_ATTEMPTS} recovery attempts: "
                    f"{state.plan_scope_revision_reason or 'revision required'}"
                ),
                summary=summary,
            )
        artifact = memory.plan_artifact
        self._append_system_feedback(
            state,
            memory,
            {
                "error": "plan_scope_revision_required",
                "reason": state.plan_scope_revision_reason,
                "current_plan": (
                    artifact.as_draft().model_dump(mode="json") if artifact is not None else None
                ),
                "completed_step_ids": (
                    [
                        step.step_id
                        for step in artifact.steps
                        if step.status == PlanStepStatus.COMPLETED
                    ]
                    if artifact is not None
                    else []
                ),
                "allowed_next_actions": sorted(_PLAN_SCOPE_RECOVERY_TOOLS),
                "instruction": (
                    "Use read-only tools first when a current snapshot or additional evidence "
                    "is required. Then call update_plan with the complete replacement plan, "
                    "including the newly required target while preserving completed work and "
                    "verification requirements. Do not call record_decision or retry the "
                    "rejected action. The revised plan returns to user review before execution."
                ),
            },
        )
        self._save_memory(memory, state, status=self._active_memory_status(state))
        return None

    def _tool_schemas(
        self,
        workflow_mode: WorkflowMode,
        *,
        state: SessionState | None = None,
    ) -> list[dict[str, Any]]:
        schemas = self.tool_registry.schemas()
        if workflow_mode == WorkflowMode.PLANNING:
            schemas = [
                schema
                for schema in schemas
                if schema.get("function", {}).get("name")
                in PLANNING_TOOL_NAMES | PLANNING_AUXILIARY_TOOL_NAMES
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
        if state is not None and state.structured_tool_recovery is not None:
            schemas = self._structured_tool_recovery_schemas(schemas, state)
        if self.reasoning_manager.config.enabled:
            return schemas
        return [
            schema
            for schema in schemas
            if schema.get("function", {}).get("name") != "record_decision"
        ]

    @staticmethod
    def _structured_tool_recovery_schemas(
        schemas: list[dict[str, Any]],
        state: SessionState,
    ) -> list[dict[str, Any]]:
        if state.structured_tool_recovery_requires_read:
            return [
                schema
                for schema in schemas
                if schema.get("function", {}).get("name") == "read_file"
            ]

        constrained = copy.deepcopy(schemas)
        for schema in constrained:
            function = schema.get("function", {})
            if function.get("name") != state.structured_tool_recovery:
                continue
            parameters = function.get("parameters", {})
            properties = parameters.get("properties", {})
            if state.structured_tool_recovery == "edit_file":
                function["description"] = (
                    "RECOVERY MODE: issue exactly one snapshot-bound line operation with at "
                    f"most {RECOVERY_MAX_NEW_LINES} new_lines entries. Split larger changes "
                    "across later calls and use the fresh snapshot returned after each edit."
                )
                operations = properties.get("operations", {})
                operations["maxItems"] = RECOVERY_MAX_EDIT_OPERATIONS
                for definition in parameters.get("$defs", {}).values():
                    new_lines = definition.get("properties", {}).get("new_lines")
                    if isinstance(new_lines, dict):
                        new_lines["maxItems"] = RECOVERY_MAX_NEW_LINES
            elif state.structured_tool_recovery == "write_file":
                function["description"] = (
                    "RECOVERY MODE: create only a minimal valid file. Keep content under "
                    f"{_RECOVERY_MAX_WRITE_CHARS} characters, then expand it with read_file "
                    "and snapshot-bound edit_file calls."
                )
                content = properties.get("content")
                if isinstance(content, dict):
                    content["maxLength"] = _RECOVERY_MAX_WRITE_CHARS
        return constrained

    @staticmethod
    def _structured_tool_recovery_error(
        call: ToolCall,
        state: SessionState,
    ) -> str | None:
        recovery_tool = state.structured_tool_recovery
        if recovery_tool is None:
            return None
        if state.structured_tool_recovery_requires_read:
            if call.name == "read_file":
                return None
            return (
                "Structured edit recovery requires one read_file call before another "
                f"{recovery_tool} request; received {call.name}."
            )
        if call.name != recovery_tool:
            return None
        if recovery_tool == "write_file":
            content = call.arguments.get("content")
            if isinstance(content, str) and len(content) > _RECOVERY_MAX_WRITE_CHARS:
                return (
                    "Structured write recovery accepts at most "
                    f"{_RECOVERY_MAX_WRITE_CHARS} content characters; create a minimal file "
                    "and expand it through snapshot-bound edits."
                )
            return None

        operations = call.arguments.get("operations")
        if not isinstance(operations, list):
            return None
        if len(operations) > RECOVERY_MAX_EDIT_OPERATIONS:
            return (
                "Structured edit recovery accepts exactly one operation per request; "
                "split the change across fresh snapshots."
            )
        new_line_count = sum(
            len(operation.get("new_lines", []))
            for operation in operations
            if isinstance(operation, dict) and isinstance(operation.get("new_lines", []), list)
        )
        if new_line_count > RECOVERY_MAX_NEW_LINES:
            return (
                "Structured edit recovery accepts at most "
                f"{RECOVERY_MAX_NEW_LINES} new_lines entries; edit a smaller range first."
            )
        return None

    @staticmethod
    def _plan_step_side_effect_tools(step: PlanStep | None) -> set[str]:
        """Return the only mutating capabilities valid for the active plan step."""
        if step is None:
            return set()
        return {
            PlanOperation.EDIT: {"edit_file"},
            PlanOperation.CREATE: {"write_file"},
            PlanOperation.DELETE: {"delete_path"},
            PlanOperation.COMMAND: {"run_command", "run_verification"},
            # run_command remains an input alias here because exact registered commands are
            # canonicalized to run_verification before validation and execution.
            PlanOperation.VERIFY: {"run_command", "run_verification"},
        }[step.operation]

    @staticmethod
    def _plan_step_action_contract(step: PlanStep | None) -> dict[str, Any] | None:
        """Describe the exact next mutating boundary in model-readable form."""
        if step is None:
            return None
        tool = {
            PlanOperation.EDIT: "edit_file",
            PlanOperation.CREATE: "write_file",
            PlanOperation.DELETE: "delete_path",
            PlanOperation.COMMAND: "run_command",
            PlanOperation.VERIFY: "run_verification",
        }[step.operation]
        arguments: dict[str, str] = {}
        if step.target_files:
            arguments["path"] = step.target_files[0]
        if len(step.verification_ids) == 1 and step.operation in {
            PlanOperation.COMMAND,
            PlanOperation.VERIFY,
        }:
            # Registered checks are the canonical execution form for command-shaped and
            # verification steps. A matching run_command request may still be normalized by
            # the Controller, but exposing the canonical action avoids provider ambiguity.
            tool = "run_verification"
            arguments = {"check_id": step.verification_ids[0]}
        return {
            "step_id": step.step_id,
            "tool": tool,
            "arguments": arguments,
            "one_state_changing_call_only": True,
        }

    def _completed_plan_step_for_call(
        self,
        artifact: PlanArtifact | None,
        call: ToolCall | None,
    ) -> PlanStep | None:
        """Return a completed step targeted by a stale replay, if one exists."""
        if artifact is None or call is None:
            return None
        path = call.arguments.get("path")
        if not isinstance(path, str):
            return None
        try:
            resolved = (
                self.plan_runtime.path_policy.resolve_entry(path)
                if call.name == "delete_path"
                else self.plan_runtime.path_policy.resolve(path)
            )
            normalized = resolved.relative_to(self.plan_runtime.path_policy.workspace).as_posix()
        except (OSError, ValueError):
            return None
        return next(
            (
                step
                for step in artifact.steps
                if step.status == PlanStepStatus.COMPLETED and normalized in step.target_files
            ),
            None,
        )

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
        return tool_name in {
            "delete_path",
            "edit_file",
            "run_command",
            "run_verification",
            "write_file",
        }

    def _requires_decision(self, call: ToolCall) -> bool:
        config = self.reasoning_manager.config
        if not config.enabled:
            return False
        if self._matching_active_plan(call) is not None:
            # The user-reviewed Plan Artifact already records intent, evidence, target scope,
            # risk, and expected verification for this exact action. Approval remains a
            # separate gate, so another provider-authored decision adds no authority.
            return False
        policy = self.tool_registry.decision_policy(call.name)
        if policy == DecisionPolicy.REGISTERED_PLAN:
            return False
        if policy == DecisionPolicy.APPROVAL_GATED:
            # The normalized request and approval outcome form the auditable decision. Keep
            # requiring a provider decision only in runtimes that removed the approval layer.
            return self.tool_registry.approval_engine is None
        if policy == DecisionPolicy.COMMAND:
            return config.require_for_commands
        return config.require_for_mutating_tools

    def _matching_active_plan(self, call: ToolCall) -> tuple[PlanArtifact, PlanStep] | None:
        memory = self.plan_runtime.memory
        artifact = memory.plan_artifact if memory is not None else None
        if artifact is None or artifact.status != PlanStatus.EXECUTING:
            return None
        step = artifact.current_step
        if step is None or self.plan_runtime.validate_action(call) is not None:
            return None
        return artifact, self.plan_runtime.matching_step(call) or step

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
            compaction_count = memory.compaction_count
            message_count = len(memory.messages)
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
            finally:
                if memory.compaction_count > compaction_count:
                    self._log(
                        "context_auto_compaction",
                        pressure=self.context_manager.last_pressure,
                        removed_messages=max(0, message_count - len(memory.messages)),
                        remaining_messages=len(memory.messages),
                        compaction_count=memory.compaction_count,
                    )

            protocol_checkpoints = 0
            options = self._model_request_options(state)
            required_tool = self._required_model_tool(state)
            if required_tool is not None:
                options = replace(
                    options or ModelRequestOptions(),
                    required_tool=required_tool,
                )
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
            selected_reasoning_effort = options.reasoning_effort if options else None
            self._log(
                "model_call",
                step=state.step_count + 1,
                message_count=len(messages),
                context_recovery_attempt=attempt,
                raw_estimated_prompt_tokens=estimate.raw,
                effective_estimated_prompt_tokens=estimate.effective,
                reasoning_effort=selected_reasoning_effort,
                reasoning_policy=self._reasoning_policy_mode.value,
                reasoning_phase=state.current_reasoning_phase,
                reasoning_reason=state.current_reasoning_reason,
                reasoning_effort_ceiling=getattr(
                    self.model_client,
                    "reasoning_effort_ceiling",
                    None,
                ),
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

    def _model_request_options(self, state: SessionState) -> ModelRequestOptions | None:
        if state.force_thinking_disabled:
            self._record_reasoning_selection(
                state,
                phase="recovering",
                reason="provider reasoning is disabled for protocol recovery",
            )
            return ModelRequestOptions(thinking_enabled=False)
        if state.provider_reasoning_detected and state.consecutive_length_responses >= 2:
            self._record_reasoning_selection(
                state,
                phase="recovering",
                reason="repeated truncated reasoning requires a thinking-disabled restart",
            )
            return ModelRequestOptions(thinking_enabled=False)
        if state.provider_reasoning_detected and state.consecutive_length_responses >= 1:
            self._record_reasoning_selection(
                state,
                phase="recovering",
                reason="a truncated response is retried once with a low one-call lease",
            )
            return ModelRequestOptions(reasoning_effort="low")
        ceiling = getattr(self.model_client, "reasoning_effort_ceiling", None)
        if ceiling in {"low", "medium", "high", "xhigh", "max"}:
            lease = self.reasoning_policy.choose(
                self._reasoning_context(state),
                ReasoningPolicySettings(
                    mode=self._reasoning_policy_mode,
                    floor=self.reasoning_manager.config.effort_floor,
                    ceiling=ceiling,
                    max_calls_per_run=self.reasoning_manager.config.max_calls_per_run,
                    xhigh_calls_per_run=self.reasoning_manager.config.xhigh_calls_per_run,
                ),
                ReasoningUsage(
                    current_step=state.step_count + 1,
                    max_calls=state.reasoning_max_calls,
                    xhigh_calls=state.reasoning_xhigh_calls,
                ),
            )
            if lease.effort == "max":
                state.reasoning_max_calls += 1
            elif lease.effort == "xhigh":
                state.reasoning_xhigh_calls += 1
            self._record_reasoning_selection(
                state,
                phase=lease.phase,
                reason=lease.reason,
            )
            return ModelRequestOptions(reasoning_effort=lease.effort)
        self._record_reasoning_selection(
            state,
            phase="discovering",
            reason="the selected model does not expose configurable reasoning effort",
        )
        return None

    @staticmethod
    def _required_model_tool(state: SessionState) -> str | None:
        """Force protocol recovery at the provider boundary, not only in the prompt."""
        if state.verification_plan_revision_required or (
            state.verification_plan_recovery_attempts > 0 and state.verification_plan is None
        ):
            return "register_verification"
        return None

    def _reasoning_context(self, state: SessionState) -> ReasoningContext:
        memory = self.plan_runtime.memory
        artifact = memory.plan_artifact if memory is not None else None
        active_plan = artifact is not None and artifact.status == PlanStatus.EXECUTING
        current_step = artifact.current_step if active_plan and artifact is not None else None
        failed_checks = sum(
            result.workspace_revision == state.workspace_revision
            and result.status in {VerificationStatus.FAILED, VerificationStatus.ERROR}
            for result in state.verification_results.values()
        )
        recovery = (
            state.verification_plan_recovery_attempts > 0
            or state.verification_plan_revision_required
            or state.plan_scope_revision_required
            or state.structured_tool_recovery is not None
            or failed_checks > 0
        )
        phase: ReasoningCallPhase
        if state.runtime is not None and state.runtime.phase == WorkflowPhase.FINALIZING:
            phase = "finalizing"
        elif state.workflow_mode == WorkflowMode.PLANNING:
            phase = "planning"
        elif recovery:
            phase = "recovering"
        elif active_plan:
            phase = "acting"
        elif (
            state.pending_decision is not None
            and state.pending_decision.next_action is not None
            and state.pending_decision.next_action.tool_name in {"edit_file", "write_file"}
        ):
            phase = "editing"
        else:
            phase = "discovering"

        affected_files = set(state.modified_files)
        if artifact is not None:
            affected_files.update(artifact.affected_files)
        top_level_modules = {path.split("/", 1)[0] for path in affected_files if "/" in path}
        errors = " ".join(state.recent_errors).lower()
        latest_reasoning = (
            memory.reasoning_events[-1] if memory and memory.reasoning_events else None
        )
        next_action_known = current_step is not None or (
            state.pending_decision is not None and state.pending_decision.next_action is not None
        )
        return ReasoningContext(
            phase=phase,
            affected_file_count=len(affected_files),
            unresolved_unknown_count=(
                len(latest_reasoning.open_questions) if latest_reasoning is not None else 0
            ),
            failed_hypothesis_count=max(failed_checks, min(2, state.consecutive_failures)),
            cross_module_change=(len(affected_files) >= 3 or len(top_level_modules) >= 2),
            architectural_decision=any(
                marker in state.task.lower()
                for marker in ("architecture", "architectural", "架构", "重构")
            ),
            conflicting_evidence="conflict" in errors or "矛盾" in errors,
            stale_snapshot_conflict=(
                "stale snapshot" in errors
                or "snapshot conflict" in errors
                or "snapshot_mismatch" in errors
            ),
            next_action_already_known=next_action_known,
            latency_sensitive=phase in {"discovering", "acting", "finalizing"},
        )

    def _model_phase(self, state: SessionState) -> WorkflowPhase:
        if state.runtime is not None and state.runtime.phase == WorkflowPhase.FINALIZING:
            return WorkflowPhase.FINALIZING
        if state.workflow_mode == WorkflowMode.PLANNING:
            return WorkflowPhase.PLANNING
        if (
            state.verification_plan_revision_required
            or state.plan_scope_revision_required
            or state.structured_tool_recovery is not None
            or (state.runtime is not None and state.runtime.recovery is not None)
        ):
            return WorkflowPhase.RECOVERING
        return WorkflowPhase.DECIDING

    def _change_runtime_phase(self, state: SessionState, phase: WorkflowPhase) -> None:
        if state.runtime is None:
            raise RuntimeError("runtime is not initialized")
        if state.runtime.phase != phase:
            self._dispatch_runtime_event(
                state,
                DomainEventKind.PHASE_CHANGED,
                payload={"phase": phase.value},
            )

    def _active_memory_status(self, state: SessionState) -> str:
        memory = self.plan_runtime.memory
        if state.workflow_mode == WorkflowMode.PLANNING:
            return "planning"
        if (
            memory is not None
            and memory.plan_artifact is not None
            and memory.plan_artifact.status in {PlanStatus.READY, PlanStatus.STALE}
        ):
            return "plan_ready"
        if (
            memory is not None
            and memory.plan_artifact is not None
            and memory.plan_artifact.status == PlanStatus.EXECUTING
        ):
            return "executing"
        if state.runtime is not None and state.runtime.phase == WorkflowPhase.VERIFYING:
            return "verifying"
        if state.runtime is not None and state.runtime.phase == WorkflowPhase.FINALIZING:
            return "finalizing"
        return "running"

    def _decision_epoch(self, state: SessionState) -> DecisionEpoch:
        if state.runtime is None:
            raise RuntimeError("runtime is not initialized")
        memory = self.plan_runtime.memory
        artifact = memory.plan_artifact if memory is not None else None
        current_step = (
            artifact.current_step
            if artifact is not None and artifact.status == PlanStatus.EXECUTING
            else None
        )
        verification_payload = {
            key: {
                "status": result.status.value,
                "workspace_revision": result.workspace_revision,
            }
            for key, result in sorted(state.verification_results.items())
        }
        verification_hash = hashlib.sha256(
            json.dumps(
                verification_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        revisions = state.runtime.revisions
        return DecisionEpoch(
            phase=self._model_phase(state),
            plan_step_id=current_step.step_id if current_step is not None else None,
            workspace_revision=revisions.workspace,
            knowledge_revision=revisions.knowledge,
            plan_revision=revisions.plan,
            verification_plan_revision=revisions.verification_plan,
            current_action_id=state.runtime.current_action_id,
            verification_state_hash=verification_hash,
        )

    def _begin_model_call(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
    ) -> tuple[str, str]:
        phase = self._model_phase(state)
        if phase not in MODEL_PHASES:
            raise RuntimeError(f"Model calls are forbidden during {phase.value}")
        if state.runtime is None:
            raise RuntimeError("runtime is not initialized")
        if state.runtime.phase != phase:
            self._dispatch_runtime_event(
                state,
                DomainEventKind.PHASE_CHANGED,
                payload={"phase": phase.value},
            )
        epoch = self._decision_epoch(state)
        model_call_id = f"model-{uuid4().hex[:12]}"
        record = ModelCallRecord(
            model_call_id=model_call_id,
            decision_epoch_id=epoch.epoch_id,
            phase=phase,
        )
        self._dispatch_runtime_event(
            state,
            DomainEventKind.MODEL_CALL_STARTED,
            correlation_id=model_call_id,
            payload={
                "model_call": record.model_dump(mode="json"),
                "decision_epoch": epoch.model_dump(mode="json"),
            },
        )
        # Waiting for a provider response is a runtime detail. Keep the durable
        # session lifecycle at its current workflow status so session metadata
        # remains loadable while a model call is in flight.
        self._save_memory(memory, state, status=self._active_memory_status(state))
        return model_call_id, epoch.epoch_id

    def _end_model_call(
        self,
        *,
        state: SessionState,
        memory: MemoryState,
        model_call_id: str | None,
        succeeded: bool,
    ) -> None:
        if model_call_id is None or state.runtime is None:
            return
        pending = state.runtime.model_call
        if pending is None or pending.model_call_id != model_call_id:
            return
        self._dispatch_runtime_event(
            state,
            (
                DomainEventKind.MODEL_CALL_FINISHED
                if succeeded
                else DomainEventKind.MODEL_CALL_FAILED
            ),
            correlation_id=model_call_id,
        )
        self._save_memory(memory, state, status=self._active_memory_status(state))

    def _decision_epoch_matches(
        self,
        state: SessionState,
        expected_epoch_id: str | None,
    ) -> bool:
        if expected_epoch_id is None or state.runtime is None:
            return False
        return self._decision_epoch(state).epoch_id == expected_epoch_id

    @staticmethod
    def _record_reasoning_selection(
        state: SessionState,
        *,
        phase: str,
        reason: str,
    ) -> None:
        state.current_reasoning_phase = phase
        state.current_reasoning_reason = reason

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
        self._dispatch_runtime_event(
            state,
            DomainEventKind.RUN_FINISHED,
            payload={
                "status": {
                    "success": RunStatus.COMPLETED.value,
                    "plan_ready": RunStatus.PAUSED.value,
                    "incomplete": RunStatus.FAILED.value,
                    "blocked": RunStatus.BLOCKED.value,
                    "stopped": RunStatus.PAUSED.value,
                    "error": RunStatus.ERROR.value,
                }[status],
                "reason": reason,
            },
        )
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
        memory.plan_scope_revision_required = state.plan_scope_revision_required
        memory.plan_scope_revision_reason = state.plan_scope_revision_reason
        memory.plan_scope_revision_attempts = state.plan_scope_revision_attempts
        memory.candidate_final_assessment = state.candidate_final_assessment
        memory.workflow_mode = state.workflow_mode
        memory.runtime = state.runtime.model_copy(deep=True) if state.runtime is not None else None
        memory.status = status
        if self.memory_store is not None:
            self.memory_store.save_state(
                memory,
                agent_step=state.step_count,
                tool_call_count=state.tool_call_count,
                status=status,
            )

    def _dispatch_runtime_event(
        self,
        state: SessionState,
        kind: DomainEventKind,
        *,
        correlation_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if state.runtime is None:
            state.runtime = AgentRuntimeState.create("ephemeral")
        kernel = RuntimeKernel(state.runtime)
        event = kernel.dispatch(
            kind,
            correlation_id=correlation_id,
            payload=payload,
        )
        state.runtime = kernel.state
        self._log(
            "runtime_transition",
            domain_event=event.kind.value,
            event_id=event.event_id,
            event_seq=event.seq,
            state_version=kernel.state.state_version,
            run_id=kernel.state.run_id,
            run_status=kernel.state.status.value,
            phase=kernel.state.phase.value,
            correlation_id=correlation_id,
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
