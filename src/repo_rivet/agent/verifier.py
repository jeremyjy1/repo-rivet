"""Recognize verification commands and enforce verify-after-change."""

import shlex
from pathlib import Path

from repo_rivet.agent.state import SessionState
from repo_rivet.tools.base import ToolCall, ToolResult


class Verifier:
    """Record recognized checks and decide whether normal completion is allowed."""

    def observe(self, state: SessionState, call: ToolCall, result: ToolResult) -> None:
        """Record a verification attempt after its tool result enters state."""
        if call.name == "git_diff":
            self._record(state, result.ok)
            return
        if call.name != "run_command":
            return

        command = call.arguments.get("command")
        if not isinstance(command, str) or not is_verification_command(command):
            return
        metadata = result.metadata or {}
        success = result.ok and metadata.get("exit_code") == 0 and not metadata.get("timed_out")
        self._record(state, success)

    @staticmethod
    def _record(state: SessionState, success: bool) -> None:
        state.last_verification_step = state.tool_call_count
        state.last_verification_success = success

    @staticmethod
    def can_finish(state: SessionState) -> bool:
        """Require a successful verification strictly after the latest modification."""
        if not state.modified_files:
            return True
        return (
            state.last_verification_success
            and state.last_verification_step is not None
            and state.last_change_step is not None
            and state.last_verification_step > state.last_change_step
        )


def is_verification_command(command: str) -> bool:
    """Recognize common tests, builds, linters, and syntax checks."""
    try:
        arguments = shlex.split(command)
    except ValueError:
        return False
    if not arguments:
        return False

    arguments = _unwrap_runner(arguments)
    if not arguments:
        return False
    executable = Path(arguments[0]).name.lower()
    rest = [argument.lower() for argument in arguments[1:]]

    if executable in {"pytest", "ruff", "mypy", "pyright", "tsc"}:
        return True
    if executable.startswith("python"):
        return (
            len(rest) >= 2
            and rest[0] == "-m"
            and rest[1]
            in {
                "compileall",
                "py_compile",
                "pytest",
                "unittest",
            }
        )
    if executable in {"npm", "pnpm", "yarn"}:
        return any(argument in {"build", "check", "lint", "test"} for argument in rest)
    if executable == "go":
        return bool(rest) and rest[0] == "test"
    if executable == "cargo":
        return bool(rest) and rest[0] in {"build", "check", "clippy", "test"}
    if executable == "make":
        return any(argument in {"build", "check", "lint", "test"} for argument in rest)
    if executable == "git":
        return bool(rest) and rest[0] == "diff"
    return False


def _unwrap_runner(arguments: list[str]) -> list[str]:
    if len(arguments) >= 3 and Path(arguments[0]).name.lower() == "uv" and arguments[1] == "run":
        return arguments[2:]
    return arguments
