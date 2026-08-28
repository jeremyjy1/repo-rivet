from datetime import UTC, datetime

import pytest

from repo_rivet.agent.state import SessionState
from repo_rivet.agent.verifier import Verifier
from repo_rivet.verification.models import (
    CommandSpec,
    VerificationCheck,
    VerificationKind,
    VerificationOutcome,
    VerificationPlan,
    VerificationResult,
    VerificationStatus,
)


def plan() -> VerificationPlan:
    return VerificationPlan(
        plan_id="plan-1",
        checks=[
            VerificationCheck(
                check_id="build",
                title="Build project",
                kind=VerificationKind.BUILD,
                command=CommandSpec(program="builder"),
                required=True,
                provenance="model",
            )
        ],
    )


def result(
    status: VerificationStatus,
    *,
    revision: int,
) -> VerificationResult:
    now = datetime.now(UTC)
    return VerificationResult(
        check_id="build",
        status=status,
        workspace_revision=revision,
        started_at=now,
        finished_at=now,
    )


def test_modified_files_require_registered_current_passed_checks() -> None:
    state = SessionState(task="task", modified_files={"app.py"}, workspace_revision=1)
    verifier = Verifier()

    assert verifier.completion_report(state).pending == ["plan"]

    state.verification_plan = plan()
    assert verifier.completion_report(state).pending == ["build"]

    verifier.record(state, result(VerificationStatus.FAILED, revision=1))
    assert verifier.completion_report(state).failed == ["build"]

    verifier.record(state, result(VerificationStatus.PASSED, revision=1))
    assert verifier.can_finish(state)


def test_revision_mismatch_makes_a_passed_result_stale() -> None:
    state = SessionState(
        task="task",
        modified_files={"app.py"},
        workspace_revision=2,
        verification_plan=plan(),
        verification_results={"build": result(VerificationStatus.PASSED, revision=1)},
    )

    report = Verifier().completion_report(state)

    assert not report.complete
    assert report.stale == ["build"]


def test_unmodified_task_does_not_require_a_verification_plan() -> None:
    state = SessionState(task="inspect only")

    assert Verifier().can_finish(state)
    assert Verifier().outcome(state) == VerificationOutcome.NOT_APPLICABLE


def test_user_facing_verification_outcome_is_not_a_misleading_boolean() -> None:
    verifier = Verifier()
    modified_without_plan = SessionState(task="change", modified_files={"app.py"})
    failed = SessionState(
        task="change",
        modified_files={"app.py"},
        verification_plan=plan(),
        verification_results={"build": result(VerificationStatus.FAILED, revision=0)},
    )
    passed = SessionState(
        task="change",
        modified_files={"app.py"},
        verification_plan=plan(),
        verification_results={"build": result(VerificationStatus.PASSED, revision=0)},
    )

    assert verifier.outcome(modified_without_plan) == VerificationOutcome.NOT_RUN
    assert verifier.outcome(failed) == VerificationOutcome.FAILED
    assert verifier.outcome(passed) == VerificationOutcome.PASSED


def test_plan_requirement_may_reference_a_required_check_id_directly() -> None:
    direct = VerificationPlan(
        plan_id="plan-direct",
        requirements=["build"],
        checks=[
            VerificationCheck(
                check_id="build",
                title="Build project",
                kind=VerificationKind.BUILD,
                command=CommandSpec(program="builder"),
                required=True,
                provenance="model",
            )
        ],
    )

    assert direct.requirements == ["build"]

    with pytest.raises(ValueError, match="not covered by required checks: unknown"):
        VerificationPlan(
            plan_id="plan-invalid",
            requirements=["unknown"],
            checks=direct.checks,
        )
