"""Serializable plan artifacts and execution progress."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repo_rivet.skills.models import ActiveSkillPin


class WorkflowMode(StrEnum):
    EXECUTE = "execute"
    PLANNING = "planning"


class PlanStatus(StrEnum):
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    STALE = "stale"
    CANCELLED = "cancelled"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class PlanOperation(StrEnum):
    EDIT = "edit"
    CREATE = "create"
    DELETE = "delete"
    COMMAND = "command"
    VERIFY = "verify"


class PlanVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    title: str = Field(min_length=1, max_length=200)
    success_criteria: str = Field(min_length=1, max_length=500)


class PlanStepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
    title: str = Field(min_length=1, max_length=200)
    intent: str = Field(min_length=1, max_length=500)
    evidence_refs: list[str] = Field(min_length=1, max_length=50)
    operation: PlanOperation
    target_files: list[str] = Field(
        default_factory=list,
        max_length=50,
        description=(
            "Exact workspace paths affected by this step. Create steps represent one new file; "
            "delete steps represent one existing file or directory; parent directories are "
            "implicit and a create target must not be repeated. Created or edited files may "
            "receive bounded corrective edit calls while the plan remains in its edit phase."
        ),
    )
    verification_ids: list[str] = Field(default_factory=list, max_length=50)
    risk: str = Field(pattern=r"^(low|medium|high)$")

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "PlanStepSpec":
        if (
            self.operation in {PlanOperation.EDIT, PlanOperation.CREATE, PlanOperation.DELETE}
            and len(self.target_files) != 1
        ):
            raise ValueError(f"{self.operation.value} steps require exactly one target path")
        if self.operation == PlanOperation.VERIFY and len(self.verification_ids) != 1:
            raise ValueError("verify steps require exactly one verification ID")
        return self


class PlanStep(PlanStepSpec):
    status: PlanStepStatus = PlanStepStatus.PENDING
    last_observation_ref: str | None = None
    last_error: str | None = Field(default=None, max_length=1_000)


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=1_000)
    constraints: list[str] = Field(default_factory=list, max_length=50)
    assumptions: list[str] = Field(default_factory=list, max_length=50)
    evidence_refs: list[str] = Field(min_length=1, max_length=100)
    steps: list[PlanStepSpec] = Field(min_length=1, max_length=100)
    verification: list[PlanVerification] = Field(min_length=1, max_length=100)
    risks: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def validate_graph_and_references(self) -> "PlanDraft":
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("plan step IDs must be unique")
        verification_ids = [item.check_id for item in self.verification]
        if len(verification_ids) != len(set(verification_ids)):
            raise ValueError("plan verification IDs must be unique")
        known_verification = set(verification_ids)
        referenced_verification: set[str] = set()
        executable_verification: set[str] = set()
        for step in self.steps:
            missing = set(step.verification_ids) - known_verification
            if missing:
                raise ValueError(
                    f"step {step.step_id} references unknown verification: "
                    f"{', '.join(sorted(missing))}"
                )
            referenced_verification.update(step.verification_ids)
            if step.operation == PlanOperation.VERIFY:
                executable_verification.update(step.verification_ids)
        if referenced_verification != known_verification:
            missing = known_verification - referenced_verification
            raise ValueError(
                "verification checks are not referenced by plan steps: "
                + ", ".join(sorted(missing))
            )
        if executable_verification != known_verification:
            missing = known_verification - executable_verification
            raise ValueError(
                "verification checks lack executable verify steps: " + ", ".join(sorted(missing))
            )
        return self


class PlanArtifact(PlanDraft):
    steps: list[PlanStep] = Field(min_length=1, max_length=100)
    plan_id: str
    artifact_revision: int = Field(default=1, ge=1)
    status: PlanStatus = PlanStatus.READY
    affected_files: list[str] = Field(default_factory=list, max_length=200)
    workspace_revision: int = Field(ge=0)
    snapshots: dict[str, str] = Field(default_factory=dict)
    execution_workspace_revision: int | None = Field(default=None, ge=0)
    execution_snapshots: dict[str, str] = Field(default_factory=dict)
    system_skills: list[ActiveSkillPin] = Field(default_factory=list)
    skill: ActiveSkillPin | None = None

    @property
    def current_step(self) -> PlanStep | None:
        return next(
            (
                step
                for step in self.steps
                if step.status
                in {
                    PlanStepStatus.PENDING,
                    PlanStepStatus.RUNNING,
                    PlanStepStatus.BLOCKED,
                    PlanStepStatus.FAILED,
                }
            ),
            None,
        )

    def as_draft(self) -> PlanDraft:
        """Project controller-owned execution state out of the model-editable plan."""
        payload = self.model_dump(
            mode="json",
            include=set(PlanDraft.model_fields) - {"steps"},
        )
        payload["steps"] = [
            step.model_dump(mode="json", include=set(PlanStepSpec.model_fields))
            for step in self.steps
        ]
        return PlanDraft.model_validate(payload)
