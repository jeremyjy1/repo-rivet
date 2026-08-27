import shlex
import sys
from pathlib import Path

from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.shell import RunCommandTool, _truncate_lines


def create_tool(workspace: Path) -> RunCommandTool:
    return RunCommandTool(WorkspacePathPolicy(workspace), CommandPolicy())


def python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def test_run_command_captures_output_and_exit_code(tmp_path: Path) -> None:
    result = create_tool(tmp_path).execute({"command": python_command("print('hello')")})

    assert result.ok
    assert "Exit code: 0" in result.output
    assert "hello" in result.output
    assert result.metadata and result.metadata["exit_code"] == 0


def test_nonzero_exit_is_a_normal_observation(tmp_path: Path) -> None:
    result = create_tool(tmp_path).execute({"command": python_command("raise SystemExit(7)")})

    assert result.ok
    assert result.metadata and result.metadata["exit_code"] == 7


def test_run_command_times_out(tmp_path: Path) -> None:
    result = create_tool(tmp_path).execute(
        {
            "command": python_command("import time; time.sleep(2)"),
            "timeout_seconds": 0.05,
        }
    )

    assert not result.ok
    assert result.metadata and result.metadata["timed_out"] is True
    assert "timed out" in (result.error or "")


def test_run_command_rejects_cwd_escape(tmp_path: Path) -> None:
    result = create_tool(tmp_path).execute({"command": "pytest", "cwd": ".."})

    assert not result.ok
    assert "escapes workspace" in (result.error or "")


def test_long_output_keeps_exact_head_and_error_tail() -> None:
    lines = [f"line-{index}" for index in range(250)]

    output, truncated = _truncate_lines("\n".join(lines))
    kept = output.splitlines()

    assert truncated
    assert kept[:80] == lines[:80]
    assert kept[80] == "... (50 lines omitted) ..."
    assert kept[81:] == lines[-120:]
    assert kept[-1] == "line-249"
