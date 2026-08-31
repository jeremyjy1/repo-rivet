"""Local validation, version binding, and execution progress for plan artifacts."""

import hashlib
import json
import stat
from pathlib import Path
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
        "semantic_query",
        "git_status",
        "git_diff",
        "submit_plan",
        "update_plan",
    }
)

PLANNING_AUXILIARY_TOOL_NAMES = frozenset({"delegate_task"})


class PlanRuntime:
    def __init__(self, path_policy: WorkspacePathPolicy) -> None:
        self.path_policy = path_policy
        self.memory: MemoryState | None = None

    def bind(self, memory: MemoryState) -> None:
        self.memory = memory

    @staticmethod
    def ensure_tool_allowed(tool_name: str) -> None:
        if tool_name not in PLANNING_TOOL_NAMES | PLANNING_AUXILIARY_TOOL_NAMES:
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
        previous_plan_id = previous.plan_id if previous is not None else None
        previous_revision = previous.artifact_revision if previous is not None else 0
        affected_files, snapshots = self._bind_files(draft)
        artifact = PlanArtifact(
            **draft.model_dump(exclude={"steps"}),
            steps=self._merge_step_progress(
                draft,
                previous=previous if is_update else None,
            ),
            plan_id=previous_plan_id
            if is_update and previous_plan_id
            else f"plan-{uuid4().hex[:12]}",
            artifact_revision=(previous_revision + 1 if is_update else 1),
            status=PlanStatus.READY,
            affected_files=affected_files,
            workspace_revision=memory.workspace_revision,
            snapshots=snapshots,
            system_skills=list(memory.system_skills),
            skill=memory.active_skill,
        )
        memory.plan_artifact = artifact
        memory.workflow_mode = WorkflowMode.EXECUTE
        memory.plan_update_reason = update_reason
        if memory.runtime is not None:
            memory.runtime.revisions.plan += 1
        return artifact

    def approve(self) -> PlanArtifact:
        memory = self._memory()
        artifact = self._require_plan()
        stale_reasons = self.stale_reasons(artifact)
        if stale_reasons:
            artifact.status = PlanStatus.STALE
            raise ValueError("Plan is stale: " + "; ".join(stale_reasons))
        artifact.status = PlanStatus.EXECUTING
        if artifact.execution_workspace_revision is None:
            artifact.execution_workspace_revision = memory.workspace_revision
        if not artifact.execution_snapshots:
            artifact.execution_snapshots = dict(artifact.snapshots)
        for step in artifact.steps:
            if step.status not in {
                PlanStepStatus.COMPLETED,
            }:
                step.status = PlanStepStatus.PENDING
                step.last_error = None
        memory.workflow_mode = WorkflowMode.EXECUTE
        memory.working.pending_actions.clear()
        if memory.runtime is not None:
            memory.runtime.revisions.plan += 1
        return artifact

    def cancel(self) -> None:
        memory = self._memory()
        if memory.plan_artifact is not None:
            memory.plan_artifact.status = PlanStatus.CANCELLED
        memory.workflow_mode = WorkflowMode.EXECUTE
        if memory.runtime is not None:
            memory.runtime.revisions.plan += 1

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
            resolved = self.path_policy.resolve_entry(path)
            if not resolved.exists() and not resolved.is_symlink():
                reasons.append(f"{path} no longer exists")
                continue
            try:
                actual = _path_revision(resolved, path)
            except (OSError, ValueError) as error:
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
            PlanOperation.DELETE: {"delete_path"},
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
            resolved = (
                self.path_policy.resolve_entry(path)
                if step.operation == PlanOperation.DELETE
                else self.path_policy.resolve(path)
            )
            normalized = resolved.relative_to(self.path_policy.workspace).as_posix()
            if normalized not in step.target_files:
                if self._completed_file_refinement_step(call, normalized) is not None:
                    return None
                return f"{normalized} is outside current plan step {step.step_id}."
        if call.name == "run_verification":
            check_id = call.arguments.get("check_id")
            if check_id not in step.verification_ids:
                return f"verification {check_id} is outside current plan step {step.step_id}."
        return None

    def requires_scope_revision(self, call: ToolCall) -> bool:
        """Return whether an invalid action expands the user-approved plan boundary."""
        artifact = self._require_plan()
        step = artifact.current_step
        if step is None:
            return False
        expected_file_tool = {
            PlanOperation.EDIT: "edit_file",
            PlanOperation.CREATE: "write_file",
            PlanOperation.DELETE: "delete_path",
        }.get(step.operation)
        path = call.arguments.get("path")
        if isinstance(path, str):
            if call.name != expected_file_tool:
                return False
            resolved = (
                self.path_policy.resolve_entry(path)
                if call.name == "delete_path"
                else self.path_policy.resolve(path)
            )
            normalized = resolved.relative_to(self.path_policy.workspace).as_posix()
            return normalized not in artifact.affected_files
        if call.name == "run_verification":
            check_id = call.arguments.get("check_id")
            return isinstance(check_id, str) and check_id not in {
                check.check_id for check in artifact.verification
            }
        return False

    def matching_step(self, call: ToolCall) -> PlanStep | None:
        """Return the approved step that authorizes an already validated action."""
        artifact = self._require_plan()
        step = artifact.current_step
        if step is None:
            return None
        path = call.arguments.get("path")
        if isinstance(path, str) and step.target_files:
            resolved = (
                self.path_policy.resolve_entry(path)
                if step.operation == PlanOperation.DELETE
                else self.path_policy.resolve(path)
            )
            normalized = resolved.relative_to(self.path_policy.workspace).as_posix()
            if normalized not in step.target_files:
                return self._completed_file_refinement_step(call, normalized)
        return step

    def start_action(self, call: ToolCall | None = None) -> PlanStep | None:
        step = self.matching_step(call) if call is not None else self._require_plan().current_step
        if step is not None and step.status not in {
            PlanStepStatus.COMPLETED,
        }:
            step.status = PlanStepStatus.RUNNING
        return step

    def observe_action(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        evidence_ref: str | None,
    ) -> None:
        artifact = self._require_plan()
        step = self.matching_step(call)
        if step is None:
            return
        refinement = step.status == PlanStepStatus.COMPLETED
        if not refinement and step.status != PlanStepStatus.RUNNING:
            return
        passed = result.ok
        metadata = result.metadata or {}
        if step.operation in {PlanOperation.EDIT, PlanOperation.CREATE, PlanOperation.DELETE}:
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
            if not refinement:
                step.status = PlanStepStatus.COMPLETED
            step.last_error = None
            revision = metadata.get("workspace_revision")
            if isinstance(revision, int):
                artifact.execution_workspace_revision = revision
            if step.operation in {PlanOperation.EDIT, PlanOperation.CREATE, PlanOperation.DELETE}:
                path = metadata.get("path")
                if step.operation == PlanOperation.DELETE and isinstance(path, str):
                    artifact.execution_snapshots.pop(path, None)
                else:
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
            if not refinement:
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

    def _completed_file_refinement_step(
        self,
        call: ToolCall,
        normalized_path: str,
    ) -> PlanStep | None:
        """Allow a bounded follow-up edit without reopening broader plan scope."""
        artifact = self._require_plan()
        current = artifact.current_step
        if call.name != "edit_file" or current is None or current.operation != PlanOperation.EDIT:
            return None
        return next(
            (
                step
                for step in artifact.steps
                if step.status == PlanStepStatus.COMPLETED
                and step.operation in {PlanOperation.CREATE, PlanOperation.EDIT}
                and normalized_path in step.target_files
            ),
            None,
        )

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
        create_steps: dict[str, str] = {}
        for step in draft.steps:
            normalized_targets: list[str] = []
            for target in step.target_files:
                resolved = (
                    self.path_policy.resolve_entry(target)
                    if step.operation == PlanOperation.DELETE
                    else self.path_policy.resolve(target)
                )
                normalized = resolved.relative_to(self.path_policy.workspace).as_posix()
                normalized_targets.append(normalized)
                if normalized not in affected:
                    affected.append(normalized)
                if resolved.exists() or resolved.is_symlink():
                    if step.operation == PlanOperation.DELETE and resolved.is_dir():
                        if not self._directory_was_inspected(step, normalized):
                            raise ValueError(
                                f"target directory was not inspected with list_files: {normalized}"
                            )
                        snapshots[normalized] = _path_revision(resolved, normalized)
                        continue
                    if (
                        step.operation in {PlanOperation.EDIT, PlanOperation.DELETE}
                        and normalized in memory.invalidated_files
                    ):
                        raise ValueError(
                            f"target file must be reread after its last change: {normalized}"
                        )
                    snapshot_id = memory.current_snapshots.get(normalized)
                    if snapshot_id is None:
                        raise ValueError(f"target file was not inspected: {normalized}")
                    current = _path_revision(resolved, normalized)
                    if current != snapshot_id:
                        raise ValueError(
                            f"target file changed; reread before planning: {normalized}"
                        )
                    snapshots[normalized] = snapshot_id
                elif step.operation != PlanOperation.CREATE:
                    raise ValueError(f"target file does not exist: {normalized}")
            step.target_files = normalized_targets
            if step.operation == PlanOperation.CREATE:
                target = normalized_targets[0]
                previous_step = create_steps.get(target)
                if previous_step is not None:
                    raise ValueError(
                        f"create target {target} is repeated by steps "
                        f"{previous_step} and {step.step_id}; write_file creates parent "
                        "directories automatically, so each new file needs only one create step"
                    )
                create_steps[target] = step.step_id
        return affected, snapshots

    def _directory_was_inspected(self, step: PlanStepSpec, normalized: str) -> bool:
        events = {event.event_id: event for event in self._memory().observation_events}
        for reference in step.evidence_refs:
            event = events.get(reference)
            if event is None or not event.ok or event.tool_name != "list_files":
                continue
            for affected in event.affected_paths:
                try:
                    inspected = self.path_policy.relative(affected).as_posix()
                except ValueError:
                    continue
                if inspected == normalized:
                    return True
        return False

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
                and previous is not None
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

    def _require_plan(self) -> PlanArtifact:
        plan = self._memory().plan_artifact
        if plan is None:
            raise ValueError("No plan artifact is available")
        return plan

    def _memory(self) -> MemoryState:
        if self.memory is None:
            raise RuntimeError("Plan runtime is not bound to session memory")
        return self.memory


def _path_revision(path: Path, relative_path: str) -> str:
    """Return a stable revision for a text file, symlink, or directory tree."""
    if path.is_file() and not path.is_symlink():
        return TextDocument.load(path).to_snapshot(relative_path=relative_path).snapshot_id

    digest = hashlib.sha256()
    pending = [path]
    while pending:
        current = pending.pop()
        current_stat = current.lstat()
        relative = "." if current == path else current.relative_to(path).as_posix()
        digest.update(
            json.dumps(
                [
                    relative,
                    stat.S_IFMT(current_stat.st_mode),
                    current_stat.st_size,
                    current_stat.st_mtime_ns,
                    current_stat.st_ino,
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if stat.S_ISDIR(current_stat.st_mode) and not stat.S_ISLNK(current_stat.st_mode):
            pending.extend(sorted(current.iterdir(), key=lambda item: item.name, reverse=True))
    return digest.hexdigest()
