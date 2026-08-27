import pytest

from repo_rivet.agent.state import SessionState
from repo_rivet.agent.verifier import Verifier, is_verification_command
from repo_rivet.tools.base import ToolCall, ToolResult


def record(state: SessionState, verifier: Verifier, call: ToolCall, result: ToolResult) -> None:
    state.record_tool_result(call, result)
    verifier.observe(state, call, result)


def test_modified_files_require_successful_later_verification() -> None:
    state = SessionState(task="task")
    verifier = Verifier()
    write = ToolCall(id="1", name="write_file", arguments={"path": "app.py"})
    pytest_call = ToolCall(id="2", name="run_command", arguments={"command": "pytest -q"})

    record(state, verifier, write, ToolResult(ok=True, output="written"))
    assert not verifier.can_finish(state)

    record(
        state,
        verifier,
        pytest_call,
        ToolResult(ok=True, output="failed", metadata={"exit_code": 1}),
    )
    assert not verifier.can_finish(state)

    record(
        state,
        verifier,
        pytest_call,
        ToolResult(ok=True, output="passed", metadata={"exit_code": 0}),
    )
    assert verifier.can_finish(state)


def test_later_change_invalidates_previous_verification() -> None:
    state = SessionState(task="task")
    verifier = Verifier()
    write = ToolCall(id="1", name="write_file", arguments={"path": "app.py"})
    diff = ToolCall(id="2", name="git_diff", arguments={})

    record(state, verifier, write, ToolResult(ok=True, output="written"))
    record(state, verifier, diff, ToolResult(ok=True, output="diff"))
    assert verifier.can_finish(state)

    record(state, verifier, write, ToolResult(ok=True, output="rewritten"))
    assert not verifier.can_finish(state)


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "uv run ruff check .",
        "python -m compileall src",
        "npm run build",
        "go test ./...",
        "cargo check",
    ],
)
def test_recognize_verification_commands(command: str) -> None:
    assert is_verification_command(command)


def test_do_not_treat_arbitrary_command_as_verification() -> None:
    assert not is_verification_command("python -c 'print(1)'")
