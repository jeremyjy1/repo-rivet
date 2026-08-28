"""Typed approval requests, assessments, decisions, and grants."""

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApprovalMode(StrEnum):
    """How non-hard-denied tool requests are handled."""

    ALLOW_ALL = "allow-all"
    LLM_AUTO = "llm-auto"
    SAFE_AUTO = "safe-auto"
    ALWAYS_ASK = "always-ask"
    READ_ONLY = "read-only"


class ApprovalAction(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"
    DEFER_TO_LLM = "defer_to_llm"


class ApprovalScope(StrEnum):
    ONCE = "once"
    SESSION_EXACT = "session_exact"


class NonInteractivePolicy(StrEnum):
    DENY = "deny"
    FAIL = "fail"


class RiskLevel(IntEnum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Capability(StrEnum):
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    FILESYSTEM_DELETE = "filesystem_delete"
    PROCESS_EXECUTE = "process_execute"
    NETWORK_ACCESS = "network_access"
    SECRET_READ = "secret_read"
    OUTSIDE_WORKSPACE = "outside_workspace"
    DEVICE_ACCESS = "device_access"
    GIT_WRITE = "git_write"
    GIT_HISTORY_REWRITE = "git_history_rewrite"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: RiskLevel
    obviously_safe: bool = False
    hard_denied: bool = False
    capabilities: set[Capability] = Field(default_factory=set)
    reasons: list[str] = Field(default_factory=list)
    affected_paths: list[str] = Field(default_factory=list)
    sensitive_paths: list[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    session_id: str
    tool_name: str
    arguments: dict[str, Any]
    normalized_arguments: dict[str, Any]
    declared_capabilities: set[Capability] = Field(default_factory=set)
    workspace: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    fingerprint: str
    assessment: RiskAssessment


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ApprovalAction
    source: str
    reason: str
    risk_level: RiskLevel
    request_fingerprint: str
    scope: ApprovalScope = ApprovalScope.ONCE
    expires_at: datetime | None = None
    llm_confidence: float | None = Field(default=None, ge=0, le=1)
    abort_agent: bool = False
    guidance: str | None = Field(default=None, min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_guidance(self) -> "ApprovalDecision":
        if self.guidance is not None and (self.action != ApprovalAction.DENY or self.abort_agent):
            raise ValueError("guidance is only valid for a non-aborting denial")
        return self


class ApprovalGrant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_fingerprint: str
    session_id: str
    action: Literal["allow", "deny"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    guidance: str | None = Field(default=None, min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_guidance(self) -> "ApprovalGrant":
        if self.guidance is not None and self.action != "deny":
            raise ValueError("guidance is only valid for a denial grant")
        return self


class LLMReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["allow", "ask", "deny"]
    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1)
    reason: str
    conditions: list[str] = Field(default_factory=list)
