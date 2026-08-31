"""Typed delegation contracts and durable child-run records."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunKind(StrEnum):
    PARENT = "parent"
    SUBAGENT = "subagent"


class SubagentProfile(StrEnum):
    EXPLORER = "explorer"
    TEST_ANALYST = "test_analyst"
    REVIEWER = "reviewer"


class SubagentStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    REPORT_READY = "report_ready"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    STALE = "stale"
    ERROR = "error"


class AgentRuntimeConfig(BaseModel):
    """Capabilities enforced when constructing a parent or child runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_kind: RunKind
    allowed_tools: frozenset[str]
    allowed_paths: tuple[str, ...]
    max_model_calls: int = Field(ge=1, le=20)
    max_tool_calls: int = Field(ge=1, le=50)
    max_runtime_seconds: float = Field(gt=0, le=600)
    can_modify_workspace: bool = False
    can_request_approval: bool = False
    can_spawn_subagents: bool = False
    can_ask_user: bool = False
    can_finish_parent_task: bool = False


class DelegateTaskArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: SubagentProfile
    objective: str = Field(min_length=1, max_length=500)
    deliverable: str = Field(min_length=1, max_length=500)
    scope_paths: list[str] = Field(min_length=1, max_length=8)
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    constraints: list[str] = Field(default_factory=list, max_length=20)
    join_policy: Literal["wait"] = "wait"


class DelegationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delegation_id: str
    parent_run_id: str
    profile: SubagentProfile
    objective: str
    deliverable: str
    scope_paths: list[str]
    excluded_paths: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    base_workspace_revision: int = Field(ge=0)
    base_plan_revision: int = Field(ge=0)
    max_model_calls: int = Field(default=4, ge=1, le=20)
    max_tool_calls: int = Field(default=10, ge=1, le=50)
    max_runtime_seconds: float = Field(default=90, gt=0, le=600)
    join_policy: Literal["wait"] = "wait"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=1_000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    affected_paths: list[str] = Field(default_factory=list, max_length=20)
    importance: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def require_evidence_for_material_finding(self) -> Finding:
        if self.importance in {"medium", "high"} and not self.evidence_refs:
            raise ValueError("medium and high importance findings require evidence_refs")
        return self


class SubagentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delegation_id: str
    status: Literal["completed", "blocked", "inconclusive"]
    summary: str = Field(min_length=1, max_length=2_000)
    findings: list[Finding] = Field(default_factory=list, max_length=30)
    recommended_actions: list[str] = Field(default_factory=list, max_length=20)
    unknowns: list[str] = Field(default_factory=list, max_length=20)
    files_read: list[str] = Field(default_factory=list, max_length=30)
    snapshots_used: dict[str, str] = Field(default_factory=dict)
    verification_refs: list[str] = Field(default_factory=list, max_length=20)
    base_workspace_revision: int = Field(ge=0)


class SubagentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subagent_id: str
    delegation_id: str
    semantic_key: str
    parent_run_id: str
    profile: SubagentProfile
    status: SubagentStatus
    base_workspace_revision: int = Field(ge=0)
    scope_paths: list[str]
    child_run_id: str | None = None
    report_event_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ValidatedSubagentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: SubagentReport
    freshness: Literal["fresh", "stale"]
    stale_paths: list[str] = Field(default_factory=list)
