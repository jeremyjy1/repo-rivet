"""Action proposal classification and durable record construction."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from repo_rivet.actions.identity import ActionIdentity, IdentityContext
from repo_rivet.actions.models import (
    ActionRecord,
    ActionStatus,
    DuplicateDisposition,
    RecoveryState,
)
from repo_rivet.agent.phases import RevisionVector, WorkflowPhase
from repo_rivet.agent.runtime_state import AgentRuntimeState
from repo_rivet.tools.base import ToolCall


@dataclass(frozen=True, slots=True)
class ProposalClassification:
    disposition: DuplicateDisposition
    semantic_key: str
    previous: ActionRecord | None = None


class ActionRegistry:
    def classify(
        self,
        call: ToolCall,
        *,
        runtime: AgentRuntimeState,
        context: IdentityContext,
        plan_step_id: str | None,
    ) -> ProposalClassification:
        key = ActionIdentity.build(
            call,
            context=context,
            revisions=runtime.revisions,
            plan_step_id=plan_step_id,
        )
        recovery = runtime.recovery
        if recovery is not None and key in recovery.forbidden_action_keys:
            return ProposalClassification(
                DuplicateDisposition.BLOCK,
                key,
                runtime.actions.get(recovery.failed_action_id or ""),
            )
        active = next((item for item in runtime.actions.values() if not item.terminal), None)
        if active is not None:
            return ProposalClassification(
                DuplicateDisposition.WAIT_EXISTING,
                key,
                active,
            )
        previous = next(
            (item for item in reversed(list(runtime.actions.values())) if item.semantic_key == key),
            None,
        )
        if previous is None:
            return ProposalClassification(DuplicateDisposition.EXECUTE_NEW, key)
        if previous.status in {
            ActionStatus.WAITING_APPROVAL,
            ActionStatus.DISPATCHED,
            ActionStatus.RUNNING,
        }:
            return ProposalClassification(DuplicateDisposition.WAIT_EXISTING, key, previous)
        if previous.result is not None and not previous.result_delivered_to_model:
            return ProposalClassification(
                DuplicateDisposition.REPLAY_UNDELIVERED_RESULT,
                key,
                previous,
            )
        if previous.status == ActionStatus.SUCCEEDED and ActionIdentity.result_still_valid(
            previous,
            context=context,
            revisions=runtime.revisions,
        ):
            return ProposalClassification(DuplicateDisposition.REUSE_RESULT, key, previous)
        if previous.status == ActionStatus.CANCELLED and previous.retryable:
            return ProposalClassification(DuplicateDisposition.EXECUTE_NEW, key, previous)
        if previous.status in {
            ActionStatus.FAILED,
            ActionStatus.CANCELLED,
            ActionStatus.INTERRUPTED_UNKNOWN,
        }:
            return ProposalClassification(
                DuplicateDisposition.REQUIRE_ALTERNATIVE,
                key,
                previous,
            )
        return ProposalClassification(DuplicateDisposition.EXECUTE_NEW, key, previous)

    @staticmethod
    def build_record(
        call: ToolCall,
        *,
        semantic_key: str,
        runtime: AgentRuntimeState,
        revisions: RevisionVector,
        plan_step_id: str | None,
        continuation_phase: WorkflowPhase,
    ) -> ActionRecord:
        return ActionRecord(
            action_id=f"action-{uuid4().hex[:12]}",
            semantic_key=semantic_key,
            session_id=runtime.session_id,
            run_id=runtime.run_id,
            plan_step_id=plan_step_id,
            tool_call_id=call.id,
            tool_name=call.name,
            normalized_arguments=call.arguments,
            revisions=revisions.model_copy(deep=True),
            continuation_phase=continuation_phase,
        )

    @staticmethod
    def recovery_for(
        action: ActionRecord,
        *,
        reason_code: str,
        evidence_refs: list[str] | None = None,
    ) -> RecoveryState:
        return RecoveryState(
            recovery_id=f"recovery-{uuid4().hex[:12]}",
            reason_code=reason_code,
            failed_action_id=action.action_id,
            forbidden_action_keys={action.semantic_key},
            evidence_refs=evidence_refs or [],
        )
