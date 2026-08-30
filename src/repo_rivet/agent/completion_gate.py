"""Deterministic completion gate independent from model completion claims."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from repo_rivet.agent.runtime import AgentRuntimeState
from repo_rivet.memory.models import MemoryState
from repo_rivet.planning.models import PlanStatus, PlanStepStatus


class RuntimeCompletionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    verification_complete: bool
    pending_steps: list[str] = Field(default_factory=list)
    active_action_ids: list[str] = Field(default_factory=list)
    stale_plan: bool = False


class CompletionGate:
    @staticmethod
    def evaluate(
        runtime: AgentRuntimeState,
        memory: MemoryState,
        *,
        verification_complete: bool,
    ) -> RuntimeCompletionReport:
        artifact = memory.plan_artifact
        pending_steps = (
            [step.step_id for step in artifact.steps if step.status != PlanStepStatus.COMPLETED]
            if artifact is not None
            else []
        )
        stale_plan = artifact is not None and artifact.status == PlanStatus.STALE
        active_actions = [
            action.action_id for action in runtime.actions.values() if not action.terminal
        ]
        ready = not any(
            (
                pending_steps,
                active_actions,
                stale_plan,
                not verification_complete,
            )
        )
        return RuntimeCompletionReport(
            ready=ready,
            verification_complete=verification_complete,
            pending_steps=pending_steps,
            active_action_ids=active_actions,
            stale_plan=stale_plan,
        )
