"""Timeout-aware local command execution without a shell."""

import os
import signal
import subprocess
import time
from pathlib import Path

from pydantic import Field

from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import BaseTool, ToolArguments, ToolResult

MAX_OUTPUT_LINES = 200


class RunCommandArguments(ToolArguments):
    command: str = Field(min_length=1)
    cwd: str = "."
    timeout_seconds: float = Field(default=60, gt=0, le=600)


class RunCommandTool(BaseTool[RunCommandArguments]):
    name = "run_command"
    description = "Run one shell-free command in the workspace with a timeout."
    arguments_type = RunCommandArguments

    def __init__(self, path_policy: WorkspacePathPolicy, command_policy: CommandPolicy) -> None:
        self.path_policy = path_policy
        self.command_policy = command_policy

    def run(self, arguments: RunCommandArguments) -> ToolResult:
        argv = self.command_policy.validate(arguments.command)
        cwd = self.path_policy.resolve(arguments.cwd)
        if not cwd.exists() or not cwd.is_dir():
            raise ValueError(f"Command cwd is not a directory: {arguments.cwd}")
        return run_process(argv, cwd=cwd, timeout_seconds=arguments.timeout_seconds)


def run_process(argv: tuple[str, ...], *, cwd: Path, timeout_seconds: float) -> ToolResult:
    """Execute argv and capture a bounded observation for the model."""
    started_at = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=os.name == "posix",
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process(process)
        stdout, stderr = process.communicate()

    duration_seconds = time.monotonic() - started_at
    output, truncated = _format_output(process.returncode, stdout, stderr)
    metadata = {
        "exit_code": process.returncode,
        "duration_seconds": round(duration_seconds, 3),
        "timed_out": timed_out,
        "truncated": truncated,
    }
    if timed_out:
        return ToolResult(
            ok=False,
            output=output,
            error=f"Command timed out after {timeout_seconds:g} seconds",
            metadata=metadata,
        )
    return ToolResult(ok=True, output=output, metadata=metadata)


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
    kept = [*lines[:100], f"... ({omitted} lines omitted) ...", *lines[-100:]]
    return "\n".join(kept), True
