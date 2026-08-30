"""Domain event envelopes used by the runtime kernel."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DomainEventKind(StrEnum):
    RUN_ACTIVATED = "run_activated"
    RUNTIME_RECONCILED = "runtime_reconciled"
    PHASE_CHANGED = "phase_changed"
    MODEL_CALL_STARTED = "model_call_started"
    MODEL_CALL_FINISHED = "model_call_finished"
    MODEL_CALL_FAILED = "model_call_failed"
    ACTION_PROPOSED = "action_proposed"
    ACTION_PREPARED = "action_prepared"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    ACTION_DISPATCHED = "action_dispatched"
    ACTION_RUNNING = "action_running"
    ACTION_OBSERVED = "action_observed"
    ACTION_RETRY_SCHEDULED = "action_retry_scheduled"
    OBSERVATION_APPLIED = "observation_applied"
    OBSERVATION_DELIVERED = "observation_delivered"
    RECOVERY_ENTERED = "recovery_entered"
    RECOVERY_CLEARED = "recovery_cleared"
    RUN_FINISHED = "run_finished"


class DomainEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: f"evt-{uuid4().hex[:12]}")
    seq: int = Field(ge=1)
    kind: DomainEventKind
    correlation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
