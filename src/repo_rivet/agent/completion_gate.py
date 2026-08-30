"""Deterministic completion gate independent from model completion claims."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from repo_rivet.actions.models import ActionStatus
from repo_rivet.agent.runtime_state import AgentRuntimeState
from repo_rivet.memory.models import MemoryState
from repo_rivet.planning.models import PlanStatus, PlanStepStatus
from repo_rivet.verification.models import VerificationStatus


class CompletionReportV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    pending_steps: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    pending_checks: list[str] = Field(default_factory=list)
    stale_checks: list[str] = Field(default_factory=list)
    active_action_ids: list[str] = Field(default_factory=list)
    unresolved_approvals: list[str] = Field(default_factory=list)
    stale_plan: bool = False


class CompletionGate:
    @staticmethod
    def evaluate(runtime: AgentRuntimeState, memory: MemoryState) -> CompletionReportV2:
        artifact = memory.plan_artifact
        pending_steps = (
            [
                step.step_id
                for step in artifact.steps
                if step.status
                not in {
                    PlanStepStatus.COMPLETED,
                    PlanStepStatus.SATISFIED,
                    PlanStepStatus.SKIPPED,
                }
            ]
            if artifact is not None
            else []
        )
        stale_plan = artifact is not None and artifact.status == PlanStatus.STALE
        active_actions = [
            action.action_id for action in runtime.actions.values() if not action.terminal
        ]
        unresolved_approvals = [
            action.action_id
            for action in runtime.actions.values()
            if action.status == ActionStatus.WAITING_APPROVAL
        ]
        failed: list[str] = []
        pending: list[str] = []
        stale: list[str] = []
        if memory.modified_files:
            if memory.verification_plan is None:
                pending.append("verification_plan")
            else:
                for check in memory.verification_plan.checks:
                    if not check.required:
                        continue
                    result = memory.verification_results.get(check.check_id)
                    if result is None:
                        pending.append(check.check_id)
                    elif result.workspace_revision != memory.workspace_revision or (
                        result.status == VerificationStatus.STALE
                    ):
                        stale.append(check.check_id)
                    elif result.status != VerificationStatus.PASSED:
                        failed.append(check.check_id)
        ready = not any(
            (
                pending_steps,
                failed,
                pending,
                stale,
                active_actions,
                unresolved_approvals,
                stale_plan,
            )
        )
        return CompletionReportV2(
            ready=ready,
            pending_steps=pending_steps,
            failed_checks=failed,
            pending_checks=pending,
            stale_checks=stale,
            active_action_ids=active_actions,
            unresolved_approvals=unresolved_approvals,
            stale_plan=stale_plan,
        )
