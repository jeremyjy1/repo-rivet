"""Plan-based completion gate for deterministic verification results."""

from repo_rivet.agent.state import SessionState
from repo_rivet.verification.models import (
    CompletionReport,
    VerificationOutcome,
    VerificationResult,
    VerificationStatus,
)


class Verifier:
    """Evaluate required registered checks without guessing from command text."""

    @staticmethod
    def record(state: SessionState, result: VerificationResult) -> None:
        state.record_verification_result(result)

    @staticmethod
    def completion_report(state: SessionState) -> CompletionReport:
        if state.verification_plan is None:
            return CompletionReport(
                complete=not state.modified_files,
                pending=["plan"] if state.modified_files else [],
            )

        passed: list[str] = []
        failed: list[str] = []
        pending: list[str] = []
        stale: list[str] = []
        inconclusive: list[str] = []
        errors: list[str] = []
        for check in state.verification_plan.checks:
            if not check.required:
                continue
            result = state.verification_results.get(check.check_id)
            if result is None:
                pending.append(check.check_id)
            elif result.workspace_revision != state.workspace_revision:
                stale.append(check.check_id)
            elif result.status == VerificationStatus.PASSED:
                passed.append(check.check_id)
            elif result.status == VerificationStatus.FAILED:
                failed.append(check.check_id)
            elif result.status == VerificationStatus.INCONCLUSIVE:
                inconclusive.append(check.check_id)
            elif result.status == VerificationStatus.ERROR:
                errors.append(check.check_id)
            else:
                pending.append(check.check_id)

        return CompletionReport(
            complete=not any((failed, pending, stale, inconclusive, errors)),
            passed=passed,
            failed=failed,
            pending=pending,
            stale=stale,
            inconclusive=inconclusive,
            errors=errors,
        )

    def can_finish(self, state: SessionState) -> bool:
        return self.completion_report(state).complete

    @classmethod
    def outcome(cls, state: SessionState) -> VerificationOutcome:
        report = cls.completion_report(state)
        if state.verification_plan is None:
            return (
                VerificationOutcome.NOT_RUN
                if state.modified_files
                else VerificationOutcome.NOT_APPLICABLE
            )
        if report.complete:
            return VerificationOutcome.PASSED
        if report.failed or report.inconclusive or report.errors:
            return VerificationOutcome.FAILED
        return VerificationOutcome.NOT_RUN
