from pathlib import Path

import pytest

from repo_rivet.memory.models import MemoryState
from repo_rivet.safety.command_policy import CommandPolicy
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.shell import ProcessExecution
from repo_rivet.verification.runtime import VerificationRuntime


def runtime(tmp_path: Path) -> VerificationRuntime:
    value = VerificationRuntime(WorkspacePathPolicy(tmp_path), CommandPolicy())
    value.bind(MemoryState(session_id="verification-test", workspace_revision=3))
    return value


def register(
    value: VerificationRuntime,
    *,
    kind: str,
    criteria: dict[str, object] | None = None,
) -> None:
    value.register_plan(
        {
            "checks": [
                {
                    "check_id": "check",
                    "title": "Explicit check",
                    "kind": kind,
                    "command": {"program": "g++", "args": ["quick_sort.cpp"]},
                    "criteria": criteria or {"expected_exit_codes": [0]},
                    "required": True,
                    "provenance": "model",
                }
            ]
        }
    )


def execution(
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> ProcessExecution:
    return ProcessExecution(
        argv=("g++", "quick_sort.cpp"),
        cwd=Path("/workspace"),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        duration_seconds=0.1,
    )


def test_build_passes_from_registered_criteria_and_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    value = runtime(tmp_path)
    register(
        value,
        kind="build",
        criteria={"expected_exit_codes": [0], "required_artifacts": ["quick_sort"]},
    )
    (tmp_path / "quick_sort").write_text("binary", encoding="utf-8")
    monkeypatch.setattr(
        "repo_rivet.verification.runtime.execute_process",
        lambda *args, **kwargs: execution(),
    )

    result = value.run("check")

    assert result.metadata
    assert result.metadata["verification_result"]["status"] == "passed"


def test_registered_command_match_requires_exact_argv_and_working_directory(
    tmp_path: Path,
) -> None:
    value = runtime(tmp_path)
    register(value, kind="build")
    (tmp_path / "nested").mkdir()

    assert value.command_matches(
        "check",
        command="g++ quick_sort.cpp",
        cwd=".",
    )
    assert not value.command_matches(
        "check",
        command="g++ -O2 quick_sort.cpp",
        cwd=".",
    )
    assert not value.command_matches(
        "check",
        command="g++ quick_sort.cpp",
        cwd="nested",
    )


def test_build_fails_when_exit_is_zero_but_artifact_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    value = runtime(tmp_path)
    register(
        value,
        kind="build",
        criteria={"expected_exit_codes": [0], "required_artifacts": ["quick_sort"]},
    )
    monkeypatch.setattr(
        "repo_rivet.verification.runtime.execute_process",
        lambda *args, **kwargs: execution(),
    )

    result = value.run("check")

    assert result.metadata
    assert result.metadata["verification_result"]["status"] == "failed"


def test_build_fails_when_compiler_exits_nonzero(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    value = runtime(tmp_path)
    register(value, kind="build")
    monkeypatch.setattr(
        "repo_rivet.verification.runtime.execute_process",
        lambda *args, **kwargs: execution(exit_code=1, stderr="compile error"),
    )

    result = value.run("check")

    assert result.metadata
    assert result.metadata["verification_result"]["status"] == "failed"


def test_smoke_check_can_pass_from_declared_exit_criteria(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    value = runtime(tmp_path)
    register(value, kind="smoke")
    monkeypatch.setattr(
        "repo_rivet.verification.runtime.execute_process",
        lambda *args, **kwargs: execution(),
    )

    result = value.run("check")

    assert result.metadata
    assert result.metadata["verification_result"]["status"] == "passed"


@pytest.mark.parametrize("kind", ["behavior", "custom"])
def test_registration_rejects_behavior_or_custom_check_without_oracle(
    tmp_path: Path,
    kind: str,
) -> None:
    value = runtime(tmp_path)

    with pytest.raises(ValueError, match="requires a deterministic output oracle"):
        register(value, kind=kind)


def test_behavior_output_oracle_passes_and_fails_deterministically(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    value = runtime(tmp_path)
    register(
        value,
        kind="behavior",
        criteria={"expected_exit_codes": [0], "stdout_exact": "1 2 3\n"},
    )
    monkeypatch.setattr(
        "repo_rivet.verification.runtime.execute_process",
        lambda *args, **kwargs: execution(stdout="1 2 3\n"),
    )
    passed = value.run("check")
    monkeypatch.setattr(
        "repo_rivet.verification.runtime.execute_process",
        lambda *args, **kwargs: execution(stdout="3 2 1\n"),
    )
    failed = value.run("check")

    assert passed.metadata and passed.metadata["verification_result"]["status"] == "passed"
    assert failed.metadata and failed.metadata["verification_result"]["status"] == "failed"


def test_invalid_plan_error_is_concise_and_actionable(tmp_path: Path) -> None:
    value = runtime(tmp_path)

    with pytest.raises(ValueError) as captured:
        value.register_plan(
            {
                "requirements": ["missing"],
                "checks": [
                    {
                        "check_id": "check",
                        "title": "Explicit check",
                        "kind": "test",
                        "command": {"program": "pytest"},
                        "required": True,
                        "provenance": "model",
                    }
                ],
            }
        )

    message = str(captured.value)
    assert message == (
        "Invalid verification plan: plan: verification requirements are not covered by "
        "required checks: missing"
    )
    assert "pydantic.dev" not in message
    assert "input_value" not in message
