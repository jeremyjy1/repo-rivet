"""Typed approval requests, assessments, decisions, and grants."""

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Annotated, Any, Literal

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
    task_summary: str = ""
    deterministic_effects: set[str] = Field(default_factory=set)
    available_constraints: set[str] = Field(default_factory=set)


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ApprovalAction
    source: str
    reason: str
    risk_level: RiskLevel
    request_fingerprint: str
    scope: ApprovalScope = ApprovalScope.ONCE
    expires_at: datetime | None = None
    constraints: list[str] = Field(default_factory=list)
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


ReviewEffect = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
ReviewFact = Annotated[str, Field(min_length=1, max_length=300)]


class LLMReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: Literal["allow", "ask", "deny"]
    risk_level: Literal["safe", "low", "medium", "high", "critical"]
    task_relevance: Literal["required", "helpful", "unrelated", "uncertain"]
    recognized_effects: list[ReviewEffect] = Field(max_length=50)
    unknowns: list[ReviewFact] = Field(default_factory=list, max_length=20)
    required_constraints: list[ReviewEffect] = Field(default_factory=list, max_length=20)
    reason: str = Field(min_length=1, max_length=400)
    user_prompt: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_user_prompt(self) -> "LLMReviewResult":
        if self.recommendation == "ask" and not self.user_prompt:
            raise ValueError("user_prompt is required when recommendation is ask")
        if self.recommendation != "ask" and self.user_prompt is not None:
            raise ValueError("user_prompt must be null unless recommendation is ask")
        return self
