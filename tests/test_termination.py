import pytest

from repo_rivet.agent.state import SessionState
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy


def test_termination_limits_are_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        TerminationConfig(max_steps=0)


@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [
        ("step_count", 3, "maximum agent steps"),
        ("consecutive_failures", 2, "consecutive tool failures"),
        ("consecutive_protocol_failures", 2, "decision protocol failures"),
        ("repeated_tool_calls", 2, "repeated identical tool call"),
        ("empty_model_responses", 2, "empty model responses"),
        ("consecutive_length_responses", 2, "length-limited model responses"),
    ],
)
def test_termination_counter_limits(attribute: str, value: int, reason: str) -> None:
    state = SessionState(task="task")
    setattr(state, attribute, value)
    policy = TerminationPolicy(
        TerminationConfig(
            max_steps=3,
            max_consecutive_failures=2,
            max_consecutive_protocol_failures=2,
            max_repeated_tool_calls=2,
            max_empty_model_responses=2,
            max_consecutive_length_responses=2,
        )
    )

    assert reason in (policy.check(state) or "")


def test_termination_runtime_limit() -> None:
    state = SessionState(task="task", started_at=10)
    policy = TerminationPolicy(TerminationConfig(max_seconds=5))

    assert "maximum runtime" in (policy.check(state, now=15) or "")


def test_no_termination_before_limit() -> None:
    state = SessionState(task="task", started_at=10)

    assert TerminationPolicy().check(state, now=11) is None
