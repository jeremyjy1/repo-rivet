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
from repo_rivet.agent.phases import RevisionVector
from repo_rivet.agent.runtime import AgentRuntimeState
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
        failed_action = (
            runtime.actions.get(recovery.failed_action_id or "") if recovery is not None else None
        )
        if failed_action is not None and key == failed_action.semantic_key:
            if recovery is not None and recovery.retry_semantic_key == key:
                return ProposalClassification(
                    DuplicateDisposition.EXECUTE_NEW,
                    key,
                    failed_action,
                )
            return ProposalClassification(
                DuplicateDisposition.BLOCK,
                key,
                failed_action,
            )
        previous = next(
            (item for item in reversed(list(runtime.actions.values())) if item.semantic_key == key),
            None,
        )
        if previous is None and call.name == "edit_file":
            previous = next(
                (
                    item
                    for item in reversed(list(runtime.actions.values()))
                    if ActionIdentity.applied_edit_covers(call, item, context=context)
                ),
                None,
            )
            if previous is not None:
                return ProposalClassification(DuplicateDisposition.REUSE_RESULT, key, previous)
        if previous is None:
            return ProposalClassification(DuplicateDisposition.EXECUTE_NEW, key)
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
        revisions: RevisionVector,
        plan_step_id: str | None,
    ) -> ActionRecord:
        return ActionRecord(
            action_id=f"action-{uuid4().hex[:12]}",
            semantic_key=semantic_key,
            plan_step_id=plan_step_id,
            tool_call_id=call.id,
            tool_name=call.name,
            normalized_arguments=call.arguments,
            revisions=revisions.model_copy(deep=True),
        )

    @staticmethod
    def recovery_for(
        action: ActionRecord,
        *,
        reason_code: str,
    ) -> RecoveryState:
        return RecoveryState(
            reason_code=reason_code,
            failed_action_id=action.action_id,
        )
