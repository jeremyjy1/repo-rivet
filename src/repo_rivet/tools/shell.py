"""Timeout-aware local command execution without a shell."""

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from repo_rivet.approval.models import Capability
from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import BaseTool, DecisionPolicy, ToolArguments, ToolResult
from repo_rivet.verification.models import ProcessObservation

MAX_OUTPUT_LINES = 200


@dataclass(frozen=True, slots=True)
class ProcessExecution:
    argv: tuple[str, ...]
    cwd: Path
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float


class RunCommandArguments(ToolArguments):
    command: str = Field(min_length=1)
    cwd: str = "."
    timeout_seconds: float = Field(default=60, gt=0, le=600)


class RunCommandTool(BaseTool[RunCommandArguments]):
    name = "run_command"
    description = "Run one shell-free command in the workspace with a timeout."
    arguments_type = RunCommandArguments
    capabilities = frozenset({Capability.PROCESS_EXECUTE})
    decision_policy = DecisionPolicy.COMMAND

    def __init__(self, path_policy: WorkspacePathPolicy, command_policy: CommandPolicy) -> None:
        self.path_policy = path_policy
        self.command_policy = command_policy

    def validate_arguments(
        self,
        arguments: dict[str, object],
    ) -> RunCommandArguments | ToolResult:
        validated = super().validate_arguments(arguments)
        if isinstance(validated, ToolResult):
            return validated
        try:
            self.command_policy.parse(validated.command)
        except ValueError as error:
            return ToolResult(
                ok=False,
                output="",
                error=str(error),
                error_code="invalid_command",
                retryable=False,
            )
        return validated

    def run(self, arguments: RunCommandArguments) -> ToolResult:
        argv = self.command_policy.validate(arguments.command)
        cwd = self.path_policy.resolve(arguments.cwd)
        if not cwd.exists() or not cwd.is_dir():
            raise ValueError(f"Command cwd is not a directory: {arguments.cwd}")
        try:
            return run_process(argv, cwd=cwd, timeout_seconds=arguments.timeout_seconds)
        except OSError as error:
            observation = ProcessObservation(
                command_id=f"process-{uuid4().hex[:12]}",
                argv=list(argv),
                cwd=str(cwd),
                spawn_error=str(error),
                duration_ms=0,
            )
            return ToolResult(
                ok=False,
                output="",
                error=str(error),
                error_code="io_error",
                retryable=True,
                metadata={"process_observation": observation.model_dump(mode="json")},
            )


def run_process(argv: tuple[str, ...], *, cwd: Path, timeout_seconds: float) -> ToolResult:
    """Execute argv and capture a bounded observation for the model."""
    execution = execute_process(argv, cwd=cwd, timeout_seconds=timeout_seconds)
    output, truncated = _format_output(
        execution.exit_code,
        execution.stdout,
        execution.stderr,
    )
    raw_output = _format_raw_output(
        execution.exit_code,
        execution.stdout,
        execution.stderr,
    )
    metadata = {
        "exit_code": execution.exit_code,
        "duration_seconds": round(execution.duration_seconds, 3),
        "timed_out": execution.timed_out,
        "truncated": truncated,
        "process_observation": ProcessObservation(
            command_id=f"process-{uuid4().hex[:12]}",
            argv=list(execution.argv),
            cwd=str(execution.cwd),
            exit_code=execution.exit_code,
            timed_out=execution.timed_out,
            duration_ms=max(0, round(execution.duration_seconds * 1_000)),
        ).model_dump(mode="json"),
    }
    if execution.timed_out:
        return ToolResult(
            ok=False,
            output=output,
            error=f"Command timed out after {timeout_seconds:g} seconds",
            metadata=metadata,
            raw_output=raw_output,
        )
    return ToolResult(ok=True, output=output, metadata=metadata, raw_output=raw_output)


def execute_process(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: float,
    stdin: str | None = None,
) -> ProcessExecution:
    """Execute one shell-free argv and return unclassified process facts."""
    started_at = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=os.name == "posix",
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(input=stdin, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process(process)
        stdout, stderr = process.communicate()

    duration_seconds = time.monotonic() - started_at
    return ProcessExecution(
        argv=argv,
        cwd=cwd,
        exit_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_seconds=duration_seconds,
    )


def _kill_process(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    process.kill()


def _format_output(exit_code: int | None, stdout: str, stderr: str) -> tuple[str, bool]:
    stdout_text, stdout_truncated = _truncate_lines(stdout)
    stderr_text, stderr_truncated = _truncate_lines(stderr)
    sections = [f"Exit code: {exit_code}"]
    if stdout_text:
        sections.extend(("STDOUT:", stdout_text))
    if stderr_text:
        sections.extend(("STDERR:", stderr_text))
    return "\n".join(sections), stdout_truncated or stderr_truncated


def _truncate_lines(output: str) -> tuple[str, bool]:
    lines = output.splitlines()
    if len(lines) <= MAX_OUTPUT_LINES:
        return "\n".join(lines), False
    omitted = len(lines) - MAX_OUTPUT_LINES
    kept = [*lines[:80], f"... ({omitted} lines omitted) ...", *lines[-120:]]
    return "\n".join(kept), True


def _format_raw_output(exit_code: int | None, stdout: str, stderr: str) -> str:
    sections = [f"Exit code: {exit_code}"]
    if stdout:
        sections.extend(("STDOUT:", stdout.rstrip("\n")))
    if stderr:
        sections.extend(("STDERR:", stderr.rstrip("\n")))
    return "\n".join(sections)
