"""Local evaluation for the fixed Skill requirement vocabulary."""

from __future__ import annotations

from dataclasses import dataclass

from repo_rivet.memory.models import MemoryState
from repo_rivet.planning.models import PlanStatus
from repo_rivet.verification.models import VerificationKind, VerificationStatus


@dataclass(frozen=True, slots=True)
class SkillRequirementReport:
    complete: bool
    failed: tuple[str, ...] = ()
    missing_verification_kinds: tuple[VerificationKind, ...] = ()


class SkillRequirementEvaluator:
    """Evaluate only deterministic state; Skill prose cannot define new checks."""

    def before_edit(self, memory: MemoryState, requirements: list[str]) -> list[str]:
        failed: list[str] = []
        for requirement in requirements:
            if requirement == "plan_approved" and (
                memory.plan_artifact is None or memory.plan_artifact.status != PlanStatus.EXECUTING
            ):
                failed.append(requirement)
            # Snapshot and seen-range constraints are enforced by EditingRuntime itself.
        return failed

    def before_finish(self, memory: MemoryState, requirements: list[str]) -> SkillRequirementReport:
        failed: list[str] = []
        missing_verification_kinds: list[VerificationKind] = []
        verification_kinds = {
            "required_build_passed": VerificationKind.BUILD,
            "required_tests_passed": VerificationKind.TEST,
            "required_behavior_checks_passed": VerificationKind.BEHAVIOR,
        }
        for requirement in requirements:
            kind = verification_kinds.get(requirement)
            if kind is not None:
                if not self._kind_declared(memory, kind):
                    missing_verification_kinds.append(kind)
                    failed.append(requirement)
                    continue
                if not self._kind_passed(memory, kind):
                    failed.append(requirement)
                    continue
            condition_failed = requirement == "no_stale_verification" and any(
                result.status == VerificationStatus.STALE
                or result.workspace_revision != memory.workspace_revision
                for result in memory.verification_results.values()
            )
            condition_failed = condition_failed or (
                requirement == "git_diff_reviewed" and not self._git_diff_reviewed(memory)
            )
            condition_failed = condition_failed or (
                requirement == "no_active_processes"
                and any(
                    result.exit_code is None and not result.timed_out and result.spawn_error is None
                    for result in memory.process_observations
                )
            )
            if condition_failed:
                failed.append(requirement)
        return SkillRequirementReport(
            complete=not failed,
            failed=tuple(failed),
            missing_verification_kinds=tuple(missing_verification_kinds),
        )

    @staticmethod
    def _kind_declared(memory: MemoryState, kind: VerificationKind) -> bool:
        return memory.verification_plan is not None and any(
            check.required and check.kind == kind for check in memory.verification_plan.checks
        )

    @staticmethod
    def _kind_passed(memory: MemoryState, kind: VerificationKind) -> bool:
        if memory.verification_plan is None:
            return False
        checks = [
            check
            for check in memory.verification_plan.checks
            if check.required and check.kind == kind
        ]
        return bool(checks) and all(
            (result := memory.verification_results.get(check.check_id)) is not None
            and result.status == VerificationStatus.PASSED
            and result.workspace_revision == memory.workspace_revision
            for check in checks
        )

    @staticmethod
    def _git_diff_reviewed(memory: MemoryState) -> bool:
        latest_change = max(
            (
                event.step
                for event in memory.observation_events
                if event.ok and event.tool_name in {"write_file", "edit_file"}
            ),
            default=-1,
        )
        latest_review = max(
            (
                event.step
                for event in memory.observation_events
                if event.ok and event.tool_name == "git_diff"
            ),
            default=-1,
        )
        return latest_review >= latest_change
