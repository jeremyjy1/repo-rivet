"""Orthogonal run, workflow, wait, and revision state for the v2 runtime."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(StrEnum):
    ACTIVE = "active"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"


class WorkflowPhase(StrEnum):
    BOOTSTRAPPING = "bootstrapping"
    PLANNING = "planning"
    PLAN_REVIEW = "plan_review"
    DECIDING = "deciding"
    PREPARING_ACTION = "preparing_action"
    EXECUTING_ACTION = "executing_action"
    APPLYING_OBSERVATION = "applying_observation"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    COMPLETION_CHECK = "completion_check"
    FINALIZING = "finalizing"


MODEL_PHASES = frozenset(
    {
        WorkflowPhase.PLANNING,
        WorkflowPhase.DECIDING,
        WorkflowPhase.RECOVERING,
        WorkflowPhase.FINALIZING,
    }
)


class WaitKind(StrEnum):
    MODEL_RESPONSE = "model_response"
    APPROVAL = "approval"
    TOOL_COMPLETION = "tool_completion"
    PROCESS_OUTPUT = "process_output"
    USER_INPUT = "user_input"
    RETRY_BACKOFF = "retry_backoff"


class WaitState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: WaitKind
    correlation_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deadline_at: datetime | None = None
    resume_phase: WorkflowPhase
    metadata: dict[str, Any] = Field(default_factory=dict)


class RevisionVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: int = Field(default=0, ge=0)
    knowledge: int = Field(default=0, ge=0)
    plan: int = Field(default=0, ge=0)
    verification_plan: int = Field(default=0, ge=0)
    environment: int = Field(default=0, ge=0)
    approval_policy: int = Field(default=0, ge=0)


class DecisionEpoch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: WorkflowPhase
    plan_step_id: str | None = None
    workspace_revision: int
    knowledge_revision: int
    plan_revision: int
    verification_plan_revision: int
    current_action_id: str | None = None
    verification_state_hash: str

    @property
    def epoch_id(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelCallStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_call_id: str
    decision_epoch_id: str
    phase: WorkflowPhase
    profile: str = "default"
    status: ModelCallStatus = ModelCallStatus.PENDING
    attempt: int = Field(default=1, ge=1)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    deadline_at: datetime | None = None
