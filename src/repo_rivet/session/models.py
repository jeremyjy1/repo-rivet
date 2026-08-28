"""Serializable metadata for independently addressable conversations."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SessionStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    VERIFYING = "verifying"
    FINALIZING = "finalizing"
    AWAITING_VERIFICATION_PLAN = "awaiting_verification_plan"
    PAUSED = "paused"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"
    FAILED = "failed"
    ARCHIVED = "archived"


class SessionMetadata(BaseModel):
    """Small session index record; listing does not load agent memory."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    name: str
    task_preview: str
    workspace: str
    workspace_key: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    model: str | None = None
    step: int = Field(default=0, ge=0)
    parent_session_id: str | None = None

    @property
    def short_id(self) -> str:
        return self.session_id.rsplit("-", maxsplit=1)[-1]


class ActiveSessionPointer(BaseModel):
    """The selected session for one normalized workspace."""

    model_config = ConfigDict(extra="forbid")

    workspace: str
    session_id: str
    updated_at: datetime
