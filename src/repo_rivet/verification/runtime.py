"""Runtime binding and deterministic execution for registered verification checks."""

from __future__ import annotations

import re
import shlex
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from repo_rivet.memory.models import MemoryState
from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import ToolResult
from repo_rivet.tools.shell import (
    ProcessExecution,
    _format_output,
    _format_raw_output,
    execute_process,
)
from repo_rivet.verification.models import (
    ProcessObservation,
    VerificationCheck,
    VerificationKind,
    VerificationPlan,
    VerificationResult,
    VerificationStatus,
)


class VerificationRuntime:
    """Resolve check IDs against durable memory and evaluate process evidence locally."""

    def __init__(
        self,
        path_policy: WorkspacePathPolicy,
        command_policy: CommandPolicy,
    ) -> None:
        self.path_policy = path_policy
        self.command_policy = command_policy
        self.memory: MemoryState | None = None

    def bind(self, memory: MemoryState) -> None:
        self.memory = memory

    def register_plan(self, arguments: dict[str, object]) -> VerificationPlan:
        try:
            checks_value = arguments.get("checks", [])
            checks = (
                [VerificationCheck.model_validate(item) for item in checks_value]
                if isinstance(checks_value, list)
                else []
            )
            requirements_value = arguments.get("requirements", [])
            requirements = (
                [str(item) for item in requirements_value]
                if isinstance(requirements_value, list)
                else []
            )
            plan = VerificationPlan(
                plan_id=f"verify-{uuid4().hex[:12]}",
                requirements=requirements,
                checks=checks,
            )
        except ValidationError as error:
            details: list[str] = []
            for item in error.errors(include_url=False, include_input=False):
                location = ".".join(str(part) for part in item["loc"]) or "plan"
                message = str(item["msg"])
                if message.startswith("Value error, "):
                    message = message.removeprefix("Value error, ")
                details.append(f"{location}: {message}")
            raise ValueError(f"Invalid verification plan: {'; '.join(details)}") from None
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid verification plan: {error}") from None
        for check in plan.checks:
            self._validate_check(check)
        memory = self._memory()
        memory.verification_plan = plan
        memory.verification_results.clear()
        if memory.runtime is not None:
            memory.runtime.revisions.verification_plan += 1
        return plan

    def check(self, check_id: str) -> VerificationCheck:
        plan = self._memory().verification_plan
        if plan is None:
            raise ValueError("No verification plan is registered")
        for check in plan.checks:
            if check.check_id == check_id:
                return check
        raise ValueError(f"Unknown verification check: {check_id}")

    def approval_arguments(self, check_id: str) -> dict[str, object]:
        check = self.check(check_id)
        return {
            "check_id": check.check_id,
            "command": shlex.join([check.command.program, *check.command.args]),
            "cwd": check.command.cwd,
            "stdin": check.command.stdin,
            "timeout_seconds": check.command.timeout_seconds,
        }

    def command_matches(self, check_id: str, *, command: str, cwd: str) -> bool:
        """Return whether an arbitrary command exactly names a registered check command."""
        check = self.check(check_id)
        if check.command.stdin is not None:
            return False
        try:
            requested_argv = self.command_policy.validate(command)
            registered_argv = self.command_policy.validate(
                shlex.join([check.command.program, *check.command.args])
            )
            requested_cwd = self.path_policy.resolve(cwd)
            registered_cwd = self.path_policy.resolve(check.command.cwd)
        except ValueError:
            return False
        return requested_argv == registered_argv and requested_cwd == registered_cwd

    def run(self, check_id: str) -> ToolResult:
        check = self.check(check_id)
        memory = self._memory()
        started_at = datetime.now(UTC)
        argv = self.command_policy.validate(
            shlex.join([check.command.program, *check.command.args])
        )
        cwd = self.path_policy.resolve(check.command.cwd)
        if not cwd.exists() or not cwd.is_dir():
            return self._error_result(
                check,
                started_at,
                f"Verification cwd is not a directory: {check.command.cwd}",
            )
        try:
            execution = execute_process(
                argv,
                cwd=cwd,
                timeout_seconds=check.command.timeout_seconds,
                stdin=check.command.stdin,
            )
        except OSError as error:
            return self._error_result(check, started_at, str(error))

        status, reasons = self._evaluate(check, execution.stdout, execution.stderr, execution)
        result = VerificationResult(
            check_id=check.check_id,
            status=status,
            workspace_revision=memory.workspace_revision,
            exit_code=execution.exit_code,
            reasons=reasons,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
        output, truncated = _format_output(
            execution.exit_code,
            execution.stdout,
            execution.stderr,
        )
        metadata = {
            "exit_code": execution.exit_code,
            "duration_seconds": round(execution.duration_seconds, 3),
            "timed_out": execution.timed_out,
            "truncated": truncated,
            "verification_check_id": check.check_id,
            "verification_title": check.title,
            "verification_result": result.model_dump(mode="json"),
            "command": shlex.join(list(argv)),
            "cwd": check.command.cwd,
            "process_observation": ProcessObservation(
                command_id=f"process-{uuid4().hex[:12]}",
                argv=list(execution.argv),
                cwd=str(execution.cwd),
                exit_code=execution.exit_code,
                timed_out=execution.timed_out,
                duration_ms=max(0, round(execution.duration_seconds * 1_000)),
            ).model_dump(mode="json"),
        }
        raw_output = _format_raw_output(
            execution.exit_code,
            execution.stdout,
            execution.stderr,
        )
        return ToolResult(
            ok=status not in {VerificationStatus.ERROR},
            output=output,
            error=reasons[0] if status == VerificationStatus.ERROR and reasons else None,
            metadata=metadata,
            raw_output=raw_output,
            error_code="verification_error" if status == VerificationStatus.ERROR else None,
            retryable=status == VerificationStatus.ERROR,
        )

    def _validate_check(self, check: VerificationCheck) -> None:
        self.command_policy.validate(shlex.join([check.command.program, *check.command.args]))
        self.path_policy.resolve(check.command.cwd)
        for artifact in check.criteria.required_artifacts:
            self.path_policy.resolve(artifact)
        if check.kind in {VerificationKind.BEHAVIOR, VerificationKind.CUSTOM} and not (
            check.criteria.has_output_oracle or check.criteria.required_artifacts
        ):
            raise ValueError(
                f"check {check.check_id} ({check.kind.value}) requires a deterministic output "
                "oracle or required artifact; declare stdout/stderr criteria or "
                "required_artifacts"
            )

    def _evaluate(
        self,
        check: VerificationCheck,
        stdout: str,
        stderr: str,
        execution: ProcessExecution,
    ) -> tuple[VerificationStatus, list[str]]:
        if execution.timed_out:
            return VerificationStatus.ERROR, ["process timed out"]
        if execution.exit_code not in check.criteria.expected_exit_codes:
            return (
                VerificationStatus.FAILED,
                [
                    f"exit code {execution.exit_code} was not one of "
                    f"{check.criteria.expected_exit_codes}"
                ],
            )

        failures = self._output_failures(check, stdout, stderr)
        for artifact in check.criteria.required_artifacts:
            path = self.path_policy.resolve(artifact)
            if not path.exists():
                failures.append(f"required artifact does not exist: {artifact}")
        if failures:
            return VerificationStatus.FAILED, failures
        if check.kind in {VerificationKind.BEHAVIOR, VerificationKind.CUSTOM} and not (
            check.criteria.has_output_oracle or check.criteria.required_artifacts
        ):
            return VerificationStatus.INCONCLUSIVE, [
                f"{check.kind.value} check has no output or artifact oracle"
            ]
        return VerificationStatus.PASSED, ["all registered success criteria passed"]

    @staticmethod
    def _output_failures(check: VerificationCheck, stdout: str, stderr: str) -> list[str]:
        criteria = check.criteria
        failures: list[str] = []
        if criteria.stdout_exact is not None and stdout != criteria.stdout_exact:
            failures.append("stdout did not exactly match the registered value")
        failures.extend(
            f"stdout did not contain registered value: {value}"
            for value in criteria.stdout_contains
            if value not in stdout
        )
        if criteria.stdout_regex is not None and re.search(criteria.stdout_regex, stdout) is None:
            failures.append("stdout did not match the registered regular expression")
        if criteria.stderr_exact is not None and stderr != criteria.stderr_exact:
            failures.append("stderr did not exactly match the registered value")
        failures.extend(
            f"stderr did not contain registered value: {value}"
            for value in criteria.stderr_contains
            if value not in stderr
        )
        if criteria.stderr_regex is not None and re.search(criteria.stderr_regex, stderr) is None:
            failures.append("stderr did not match the registered regular expression")
        return failures

    def _error_result(
        self,
        check: VerificationCheck,
        started_at: datetime,
        reason: str,
    ) -> ToolResult:
        result = VerificationResult(
            check_id=check.check_id,
            status=VerificationStatus.ERROR,
            workspace_revision=self._memory().workspace_revision,
            reasons=[reason],
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
        return ToolResult(
            ok=False,
            output="",
            error=reason,
            error_code="verification_error",
            retryable=True,
            metadata={
                "verification_check_id": check.check_id,
                "verification_title": check.title,
                "verification_result": result.model_dump(mode="json"),
                "process_observation": ProcessObservation(
                    command_id=f"process-{uuid4().hex[:12]}",
                    argv=[check.command.program, *check.command.args],
                    cwd=check.command.cwd,
                    spawn_error=reason,
                    duration_ms=0,
                ).model_dump(mode="json"),
            },
        )

    def _memory(self) -> MemoryState:
        if self.memory is None:
            raise ValueError("Verification runtime is not bound to a session")
        return self.memory
