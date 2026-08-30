"""Durable models for one primary action and its observation delivery."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from repo_rivet.agent.phases import RevisionVector, WorkflowPhase
from repo_rivet.tools.base import ToolResult


class ActionStatus(StrEnum):
    PROPOSED = "proposed"
    PREPARED = "prepared"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    OBSERVED = "observed"
    APPLIED = "applied"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED_UNKNOWN = "interrupted_unknown"


TERMINAL_ACTION_STATUSES = frozenset(
    {
        ActionStatus.SUCCEEDED,
        ActionStatus.FAILED,
        ActionStatus.CANCELLED,
        ActionStatus.INTERRUPTED_UNKNOWN,
    }
)


class DuplicateDisposition(StrEnum):
    EXECUTE_NEW = "execute_new"
    WAIT_EXISTING = "wait_existing"
    REUSE_RESULT = "reuse_result"
    REPLAY_UNDELIVERED_RESULT = "replay_undelivered_result"
    RETRY_INTERNAL = "retry_internal"
    REQUIRE_ALTERNATIVE = "require_alternative"
    BLOCK = "block"


class RetryClass(StrEnum):
    NONE = "none"
    TRANSIENT_INFRASTRUCTURE = "transient_infrastructure"
    BUSINESS_FAILURE = "business_failure"


class ActionResultSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] | None = None
    raw_output: str | None = None
    error_code: str | None = None
    retryable: bool | None = None

    @classmethod
    def from_result(cls, result: ToolResult) -> ActionResultSnapshot:
        return cls(
            ok=result.ok,
            output=result.output,
            error=result.error,
            metadata=result.metadata,
            # Full command output remains in the session output store. The durable action
            # record keeps only the bounded context form needed for recovery and replay.
            raw_output=None,
            error_code=result.error_code,
            retryable=result.retryable,
        )

    def to_result(self) -> ToolResult:
        return ToolResult(
            ok=self.ok,
            output=self.output,
            error=self.error,
            metadata=self.metadata,
            raw_output=self.raw_output,
            error_code=self.error_code,
            retryable=self.retryable,
        )


class ObservationDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    result_event_id: str
    persisted: bool = False
    queued_for_context: bool = False
    delivered_to_model: bool = False


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    semantic_key: str
    session_id: str
    run_id: str
    plan_step_id: str | None = None
    tool_call_id: str
    tool_name: str
    normalized_arguments: dict[str, Any]
    preconditions: list[dict[str, Any]] = Field(default_factory=list)
    expected_postconditions: list[dict[str, Any]] = Field(default_factory=list)
    revisions: RevisionVector
    status: ActionStatus = ActionStatus.PROPOSED
    attempt: int = Field(default=1, ge=1)
    retry_class: RetryClass = RetryClass.NONE
    retryable: bool = False
    result_event_id: str | None = None
    result: ActionResultSnapshot | None = None
    result_applied: bool = False
    result_delivered_to_model: bool = False
    continuation_phase: WorkflowPhase = WorkflowPhase.DECIDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_ACTION_STATUSES


class RecoveryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recovery_id: str
    reason_code: str
    failed_action_id: str | None = None
    forbidden_action_keys: set[str] = Field(default_factory=set)
    evidence_refs: list[str] = Field(default_factory=list)
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=1, ge=1)
