"""Local validation, version binding, and execution progress for plan artifacts."""

from uuid import uuid4

from repo_rivet.editing.document import TextDocument
from repo_rivet.memory.models import MemoryState
from repo_rivet.planning.errors import PlanModeViolation
from repo_rivet.planning.models import (
    PlanArtifact,
    PlanDraft,
    PlanOperation,
    PlanStatus,
    PlanStep,
    PlanStepSpec,
    PlanStepStatus,
    WorkflowMode,
)
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import ToolCall, ToolResult

PLANNING_TOOL_NAMES = frozenset(
    {
        "list_files",
        "read_file",
        "search_text",
        "git_status",
        "git_diff",
        "submit_plan",
        "update_plan",
    }
)


class PlanRuntime:
    def __init__(self, path_policy: WorkspacePathPolicy) -> None:
        self.path_policy = path_policy
        self.memory: MemoryState | None = None

    def bind(self, memory: MemoryState) -> None:
        self.memory = memory

    @staticmethod
    def ensure_tool_allowed(tool_name: str) -> None:
        if tool_name not in PLANNING_TOOL_NAMES:
            raise PlanModeViolation(tool_name)

    def submit(
        self,
        arguments: dict[str, object],
        *,
        update_reason: str | None = None,
    ) -> PlanArtifact:
        memory = self._memory()
        draft_value = arguments.get("plan")
        if not isinstance(draft_value, dict):
            raise ValueError("plan must be a structured object")
        draft = PlanDraft.model_validate(draft_value)
        self._validate_evidence(draft)
        previous = memory.plan_artifact
        if update_reason is not None and previous is None:
            raise ValueError("No existing plan is available to update")
        if (
            update_reason is None
            and previous is not None
            and previous.status not in {PlanStatus.CANCELLED, PlanStatus.COMPLETED}
        ):
            raise ValueError("A plan already exists; use update_plan to revise it")
        is_update = update_reason is not None and previous is not None
        affected_files, snapshots = self._bind_files(draft)
        artifact = PlanArtifact(
            **draft.model_dump(exclude={"steps"}),
            steps=self._merge_step_progress(
                draft,
                previous=previous if is_update else None,
            ),
            plan_id=previous.plan_id if is_update else f"plan-{uuid4().hex[:12]}",
            artifact_revision=(previous.artifact_revision + 1 if is_update else 1),
            status=PlanStatus.READY,
            affected_files=affected_files,
            workspace_revision=memory.workspace_revision,
            snapshots=snapshots,
            system_skills=list(memory.system_skills),
            skill=memory.active_skill,
        )
        memory.plan_artifact = artifact
        memory.workflow_mode = WorkflowMode.PLAN_READY
        memory.plan_update_reason = update_reason
        return artifact

    def approve(self) -> PlanArtifact:
        memory = self._memory()
        artifact = self._require_plan()
        stale_reasons = self.stale_reasons(artifact)
        if stale_reasons:
            artifact.status = PlanStatus.STALE
            memory.workflow_mode = WorkflowMode.PLAN_READY
            raise ValueError("Plan is stale: " + "; ".join(stale_reasons))
        self._recover_observed_completed_creates(artifact)
        artifact.status = PlanStatus.EXECUTING
        if artifact.execution_workspace_revision is None:
            artifact.execution_workspace_revision = memory.workspace_revision
        if not artifact.execution_snapshots:
            artifact.execution_snapshots = dict(artifact.snapshots)
        for step in artifact.steps:
            if step.status != PlanStepStatus.COMPLETED:
                step.status = PlanStepStatus.PENDING
                step.last_error = None
        memory.workflow_mode = WorkflowMode.EXECUTE
        memory.reflection_required = False
        memory.working.pending_actions.clear()
        return artifact

    def cancel(self) -> None:
        memory = self._memory()
        if memory.plan_artifact is not None:
            memory.plan_artifact.status = PlanStatus.CANCELLED
        memory.workflow_mode = WorkflowMode.EXECUTE

    def stale_reasons(self, artifact: PlanArtifact | None = None) -> list[str]:
        memory = self._memory()
        value = artifact or self._require_plan()
        reasons: list[str] = []
        if value.system_skills != memory.system_skills:
            reasons.append("system Skills changed after planning")
        if value.skill != memory.active_skill:
            reasons.append("global Skill changed after planning")
        expected_revision = (
            value.execution_workspace_revision
            if value.status == PlanStatus.EXECUTING
            and value.execution_workspace_revision is not None
            else value.workspace_revision
        )
        if memory.workspace_revision != expected_revision:
            reasons.append(
                f"workspace revision changed ({expected_revision} -> {memory.workspace_revision})"
            )
        expected_snapshots = (
            value.execution_snapshots if value.status == PlanStatus.EXECUTING else value.snapshots
        )
        for path, expected in expected_snapshots.items():
            resolved = self.path_policy.resolve(path)
            if not resolved.exists():
                reasons.append(f"{path} no longer exists")
                continue
            try:
                actual = TextDocument.load(resolved).to_snapshot(relative_path=path).snapshot_id
            except ValueError as error:
                reasons.append(f"{path} cannot be validated: {error}")
                continue
            if actual != expected:
                reasons.append(f"{path} changed after planning")
        return reasons

    def validate_action(self, call: ToolCall) -> str | None:
        artifact = self._require_plan()
        step = artifact.current_step
        if step is None:
            return "All plan steps are complete; provide the final response."
        expected_tools = {
            PlanOperation.EDIT: {"edit_file"},
            PlanOperation.CREATE: {"write_file"},
            PlanOperation.COMMAND: {"run_command"},
            PlanOperation.VERIFY: {"run_verification"},
        }[step.operation]
        if step.operation == PlanOperation.COMMAND and len(step.verification_ids) == 1:
            # A deterministic registered check is a stronger execution of a command-shaped
            # plan step. Requiring run_command here would conflict with the Controller rule
            # that build and test claims must flow through run_verification.
            expected_tools.add("run_verification")
        if call.name not in expected_tools:
            return (
                f"Current plan step {step.step_id} requires {step.operation.value}; "
                f"received {call.name}. Update the plan before changing scope."
            )
        path = call.arguments.get("path")
        if isinstance(path, str) and step.target_files:
            normalized = self.path_policy.relative(path).as_posix()
            if normalized not in step.target_files:
                return f"{normalized} is outside current plan step {step.step_id}."
        if call.name == "run_verification":
            check_id = call.arguments.get("check_id")
            if check_id not in step.verification_ids:
                return f"verification {check_id} is outside current plan step {step.step_id}."
        return None

    def start_action(self) -> None:
        step = self._require_plan().current_step
        if step is not None:
            step.status = PlanStepStatus.RUNNING

    def observe_action(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        evidence_ref: str | None,
    ) -> None:
        artifact = self._require_plan()
        step = artifact.current_step
        if step is None or step.status != PlanStepStatus.RUNNING:
            return
        passed = result.ok
        metadata = result.metadata or {}
        if step.operation in {PlanOperation.EDIT, PlanOperation.CREATE}:
            result_path = metadata.get("path")
            passed = (
                passed
                and isinstance(metadata.get("workspace_revision"), int)
                and result_path in step.target_files
            )
        elif step.operation == PlanOperation.COMMAND:
            if call.name == "run_verification":
                verification = metadata.get("verification_result")
                passed = (
                    passed
                    and isinstance(verification, dict)
                    and verification.get("status") == "passed"
                )
            else:
                passed = passed and metadata.get("exit_code") == 0
        elif step.operation == PlanOperation.VERIFY:
            verification = metadata.get("verification_result")
            passed = (
                passed and isinstance(verification, dict) and verification.get("status") == "passed"
            )
        step.last_observation_ref = evidence_ref
        if passed:
            step.status = PlanStepStatus.COMPLETED
            step.last_error = None
            revision = metadata.get("workspace_revision")
            if isinstance(revision, int):
                artifact.execution_workspace_revision = revision
            if step.operation in {PlanOperation.EDIT, PlanOperation.CREATE}:
                path = metadata.get("path")
                snapshot_id = metadata.get("new_snapshot_id") or metadata.get("snapshot_id")
                if isinstance(path, str) and isinstance(snapshot_id, str):
                    artifact.execution_snapshots[path] = snapshot_id
            if call.name == "run_verification":
                check_id = call.arguments.get("check_id")
                if isinstance(check_id, str):
                    self._complete_consecutive_verified_steps(
                        artifact,
                        check_id=check_id,
                        evidence_ref=evidence_ref,
                    )
        else:
            step.status = (
                PlanStepStatus.BLOCKED
                if result.error_code
                in {
                    "approval_denied",
                    "approval_stale",
                    "hard_policy_denied",
                }
                else PlanStepStatus.FAILED
            )
            step.last_error = result.error or "tool result did not satisfy the step"
        if all(item.status == PlanStepStatus.COMPLETED for item in artifact.steps):
            artifact.status = PlanStatus.COMPLETED

    @staticmethod
    def _complete_consecutive_verified_steps(
        artifact: PlanArtifact,
        *,
        check_id: str,
        evidence_ref: str | None,
    ) -> None:
        """Advance redundant verify steps already satisfied by the executed typed check."""
        while True:
            step = artifact.current_step
            if (
                step is None
                or step.operation != PlanOperation.VERIFY
                or step.verification_ids != [check_id]
            ):
                return
            step.status = PlanStepStatus.COMPLETED
            step.last_observation_ref = evidence_ref
            step.last_error = None

    def _validate_evidence(self, draft: PlanDraft) -> None:
        memory = self._memory()
        known = {event.event_id for event in memory.observation_events}
        known.update(event.event_id for event in memory.reasoning_events)
        refs = set(draft.evidence_refs)
        refs.update(ref for step in draft.steps for ref in step.evidence_refs)
        missing = refs - known
        if missing:
            raise ValueError("plan references unknown evidence: " + ", ".join(sorted(missing)))

    def _bind_files(self, draft: PlanDraft) -> tuple[list[str], dict[str, str]]:
        memory = self._memory()
        affected: list[str] = []
        snapshots: dict[str, str] = {}
        for step in draft.steps:
            normalized_targets: list[str] = []
            for target in step.target_files:
                normalized = self.path_policy.relative(target).as_posix()
                normalized_targets.append(normalized)
                if normalized not in affected:
                    affected.append(normalized)
                resolved = self.path_policy.resolve(normalized)
                if resolved.exists():
                    if (
                        step.operation == PlanOperation.EDIT
                        and normalized in memory.invalidated_files
                    ):
                        raise ValueError(
                            f"target file must be reread after its last change: {normalized}"
                        )
                    snapshot_id = memory.current_snapshots.get(normalized)
                    if snapshot_id is None:
                        raise ValueError(f"target file was not inspected: {normalized}")
                    current = (
                        TextDocument.load(resolved)
                        .to_snapshot(relative_path=normalized)
                        .snapshot_id
                    )
                    if current != snapshot_id:
                        raise ValueError(
                            f"target file changed; reread before planning: {normalized}"
                        )
                    snapshots[normalized] = snapshot_id
                elif step.operation != PlanOperation.CREATE:
                    raise ValueError(f"target file does not exist: {normalized}")
            step.target_files = normalized_targets
        return affected, snapshots

    @classmethod
    def _merge_step_progress(
        cls,
        draft: PlanDraft,
        *,
        previous: PlanArtifact | None,
    ) -> list[PlanStep]:
        merged: list[PlanStep] = []
        for step in draft.steps:
            previous_step = cls._matching_previous_step(step, previous)
            if (
                previous_step is not None
                and step.operation == PlanOperation.VERIFY
                and not cls._verification_specs_match(step, draft, previous)
            ):
                previous_step = None
            if previous_step is not None and previous_step.status == PlanStepStatus.COMPLETED:
                merged.append(
                    PlanStep(
                        **step.model_dump(),
                        status=PlanStepStatus.COMPLETED,
                        last_observation_ref=previous_step.last_observation_ref,
                    )
                )
                continue
            merged.append(PlanStep(**step.model_dump(), status=PlanStepStatus.PENDING))
        return merged

    @staticmethod
    def _matching_previous_step(
        step: PlanStepSpec,
        previous: PlanArtifact | None,
    ) -> PlanStep | None:
        if previous is None:
            return None
        return next(
            (
                candidate
                for candidate in previous.steps
                if candidate.model_dump(include=set(PlanStepSpec.model_fields)) == step.model_dump()
            ),
            None,
        )

    @staticmethod
    def _verification_specs_match(
        step: PlanStepSpec,
        draft: PlanDraft,
        previous: PlanArtifact,
    ) -> bool:
        current = {item.check_id: item for item in draft.verification}
        prior = {item.check_id: item for item in previous.verification}
        return all(
            current.get(check_id) == prior.get(check_id) for check_id in step.verification_ids
        )

    def _recover_observed_completed_creates(self, artifact: PlanArtifact) -> None:
        """Repair plans created after a file was already written by this session."""
        memory = self._memory()
        for step in artifact.steps:
            if step.status != PlanStepStatus.PENDING or step.operation != PlanOperation.CREATE:
                continue
            path = step.target_files[0]
            if path not in artifact.snapshots:
                continue
            observation = next(
                (
                    event
                    for event in reversed(memory.observation_events)
                    if event.ok and event.tool_name == "write_file" and path in event.affected_paths
                ),
                None,
            )
            if observation is None:
                artifact.status = PlanStatus.STALE
                memory.workflow_mode = WorkflowMode.PLAN_READY
                raise ValueError(
                    "Plan is stale: create step target already existed without a matching "
                    f"successful write observation: {path}"
                )
            step.status = PlanStepStatus.COMPLETED
            step.last_observation_ref = observation.event_id

    def _require_plan(self) -> PlanArtifact:
        plan = self._memory().plan_artifact
        if plan is None:
            raise ValueError("No plan artifact is available")
        return plan

    def _memory(self) -> MemoryState:
        if self.memory is None:
            raise RuntimeError("Plan runtime is not bound to session memory")
        return self.memory
