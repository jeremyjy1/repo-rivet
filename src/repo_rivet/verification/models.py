"""Serializable models for provider-declared, locally evaluated verification."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VerificationKind(StrEnum):
    BUILD = "build"
    TEST = "test"
    BEHAVIOR = "behavior"
    LINT = "lint"
    SMOKE = "smoke"
    CUSTOM = "custom"


class VerificationStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"
    STALE = "stale"


class VerificationOutcome(StrEnum):
    """User-facing completion evidence state for one active task scope."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    NOT_APPLICABLE = "not_applicable"


class CommandSpec(BaseModel):
    """A shell-free command registered before it can count as verification."""

    model_config = ConfigDict(extra="forbid")

    program: str = Field(min_length=1, max_length=1_000)
    args: list[str] = Field(default_factory=list, max_length=200)
    cwd: str = Field(default=".", min_length=1, max_length=1_000)
    stdin: str | None = Field(default=None, max_length=100_000)
    timeout_seconds: float = Field(default=60, gt=0, le=600)


class SuccessCriteria(BaseModel):
    """Deterministic predicates evaluated against one process observation."""

    model_config = ConfigDict(extra="forbid")

    expected_exit_codes: list[int] = Field(default_factory=lambda: [0], min_length=1)
    stdout_exact: str | None = Field(default=None, max_length=100_000)
    stdout_contains: list[str] = Field(default_factory=list, max_length=100)
    stdout_regex: str | None = Field(default=None, max_length=2_000)
    stderr_exact: str | None = Field(default=None, max_length=100_000)
    stderr_contains: list[str] = Field(default_factory=list, max_length=100)
    stderr_regex: str | None = Field(default=None, max_length=2_000)
    required_artifacts: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_patterns(self) -> SuccessCriteria:
        for label, pattern in (
            ("stdout_regex", self.stdout_regex),
            ("stderr_regex", self.stderr_regex),
        ):
            if pattern is None:
                continue
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"{label} is invalid: {error}") from None
        return self

    @property
    def has_output_oracle(self) -> bool:
        return any(
            (
                self.stdout_exact is not None,
                self.stdout_contains,
                self.stdout_regex is not None,
                self.stderr_exact is not None,
                self.stderr_contains,
                self.stderr_regex is not None,
            )
        )


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    title: str = Field(min_length=1, max_length=300)
    kind: VerificationKind
    command: CommandSpec
    criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    required: bool = True
    claim_ids: list[str] = Field(default_factory=list, max_length=100)
    provenance: str = Field(pattern=r"^(user|project_adapter|model)$")


class VerificationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    requirements: list[str] = Field(default_factory=list, max_length=100)
    checks: list[VerificationCheck] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_checks(self) -> VerificationPlan:
        identifiers = [check.check_id for check in self.checks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("verification check IDs must be unique")
        if not any(check.required for check in self.checks):
            raise ValueError("verification plan must contain at least one required check")
        if len(self.requirements) != len(set(self.requirements)):
            raise ValueError("verification requirements must be unique")
        covered_requirements = {
            requirement
            for check in self.checks
            if check.required
            for requirement in (check.check_id, *check.claim_ids)
        }
        missing_requirements = set(self.requirements) - covered_requirements
        if missing_requirements:
            raise ValueError(
                "verification requirements are not covered by required checks: "
                + ", ".join(sorted(missing_requirements))
            )
        return self


class ProcessObservation(BaseModel):
    """Objective process facts; this model never claims verification success."""

    model_config = ConfigDict(extra="forbid")

    command_id: str
    argv: list[str]
    cwd: str
    exit_code: int | None = None
    stdout_ref: str | None = None
    stderr_ref: str | None = None
    timed_out: bool = False
    spawn_error: str | None = None
    duration_ms: int = Field(ge=0)


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: VerificationStatus
    workspace_revision: int = Field(ge=0)
    exit_code: int | None = None
    reasons: list[str] = Field(default_factory=list)
    stdout_ref: str | None = None
    stderr_ref: str | None = None
    started_at: datetime
    finished_at: datetime


class CompletionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complete: bool
    passed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)
    inconclusive: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def runnable(self) -> list[str]:
        return [*self.pending, *self.stale]


class FinalAssessment(BaseModel):
    """A model-authored completion opinion, separate from local verification facts."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=2_000)
    changes: list[str] = Field(default_factory=list, max_length=100)
    claimed_completed: bool = True
    remaining_risks: list[str] = Field(default_factory=list, max_length=100)
    evidence_refs: list[str] = Field(default_factory=list, max_length=100)


class ModelErrorRecord(BaseModel):
    """Sanitized provider failure details retained for diagnosis and recovery."""

    model_config = ConfigDict(extra="forbid")

    error_type: str
    status_code: int | None = None
    error_code: str | None = None
    request_id: str | None = None
    message: str
    retryable: bool
    attempt: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    message_count: int = Field(ge=0)
    message_roles: list[str]
    pending_tool_call_ids: list[str]
    request_size_bytes: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
