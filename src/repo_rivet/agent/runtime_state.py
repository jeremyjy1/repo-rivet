"""Serializable v2 state shared by the reducer, Controller, and checkpoints."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from repo_rivet.actions.models import ActionRecord, RecoveryState
from repo_rivet.agent.phases import (
    DecisionEpoch,
    ModelCallRecord,
    RevisionVector,
    RunStatus,
    WaitState,
    WorkflowPhase,
)


class AgentRuntimeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 2
    session_id: str
    run_id: str = Field(default_factory=lambda: f"run-{uuid4().hex[:12]}")
    state_version: int = Field(default=0, ge=0)
    status: RunStatus = RunStatus.ACTIVE
    phase: WorkflowPhase = WorkflowPhase.BOOTSTRAPPING
    wait: WaitState | None = None
    revisions: RevisionVector = Field(default_factory=RevisionVector)
    current_action_id: str | None = None
    actions: dict[str, ActionRecord] = Field(default_factory=dict)
    model_call: ModelCallRecord | None = None
    decision_epoch: DecisionEpoch | None = None
    current_decision_epoch_id: str | None = None
    recovery: RecoveryState | None = None
    pending_observation_ids: list[str] = Field(default_factory=list)
    terminal_reason: str | None = None
    last_event_seq: int = Field(default=0, ge=0)

    @classmethod
    def create(cls, session_id: str, *, workspace_revision: int = 0) -> AgentRuntimeState:
        return cls(
            session_id=session_id,
            revisions=RevisionVector(workspace=workspace_revision),
        )
