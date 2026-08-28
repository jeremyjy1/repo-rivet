"""Validated records for plans, decisions, reflections, and observations."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvidenceRef = Annotated[str, Field(min_length=1, max_length=200)]
BoundedNote = Annotated[str, Field(min_length=1, max_length=500)]


class ReasoningPhase(StrEnum):
    PLAN = "plan"
    DECISION = "decision"
    REFLECTION = "reflection"
    FINAL_ASSESSMENT = "final_assessment"


class ReasoningDisplayMode(StrEnum):
    OFF = "off"
    SUMMARY = "summary"
    TRACE = "trace"


class ReasoningConfig(BaseModel):
    """Bounded policy for the provider-independent decision trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    display: ReasoningDisplayMode = ReasoningDisplayMode.SUMMARY
    max_summary_chars: int = Field(default=1_000, ge=100, le=1_000)
    recent_event_limit: int = Field(default=8, ge=2, le=50)
    require_for_mutating_tools: bool = True
    require_for_commands: bool = True
    max_reflection_only_turns: int = Field(default=2, ge=1, le=10)


class ActionIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1, max_length=100)
    argument_summary: str = Field(default="", max_length=500)
    expected_result: str = Field(min_length=1, max_length=500)


class RecordDecisionArgs(BaseModel):
    """Arguments accepted from the model by the record_decision meta tool."""

    model_config = ConfigDict(extra="forbid")

    phase: ReasoningPhase
    current_goal: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=1_000)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=20)
    assumptions: list[BoundedNote] = Field(default_factory=list, max_length=20)
    open_questions: list[BoundedNote] = Field(default_factory=list, max_length=20)
    next_tool: str | None = Field(default=None, min_length=1, max_length=100)
    next_tool_argument_summary: str | None = Field(default=None, max_length=500)
    expected_result: str | None = Field(default=None, max_length=500)
    confidence: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_next_action(self) -> "RecordDecisionArgs":
        details_present = (
            self.next_tool_argument_summary is not None or self.expected_result is not None
        )
        if self.next_tool is None and details_present:
            raise ValueError("next_tool is required when next-action details are provided")
        if self.next_tool is not None and not self.expected_result:
            raise ValueError("expected_result is required when next_tool is provided")
        return self


class ReasoningEvent(BaseModel):
    """A concise claim whose evidence and intended next action can be audited."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    session_id: str
    step: int = Field(ge=0)
    phase: ReasoningPhase
    current_goal: str
    summary: str = Field(max_length=1_000)
    evidence_refs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_action: ActionIntent | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ObservationEvent(BaseModel):
    """A deterministic local record produced from a real tool result."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    session_id: str
    step: int = Field(ge=0)
    tool_call_id: str
    tool_name: str
    ok: bool
    result_summary: str = Field(max_length=1_000)
    output_ref: str | None = None
    exit_code: int | None = None
    affected_paths: list[str] = Field(default_factory=list)
    verification: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
