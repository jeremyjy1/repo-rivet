import pytest

from repo_rivet.agent.state import SessionState
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy


def test_termination_limits_are_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        TerminationConfig(max_steps=0)


@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [
        ("step_count", 3, "agent step checkpoint"),
        ("consecutive_failures", 2, "consecutive tool failures"),
        ("consecutive_protocol_failures", 2, "action protocol failures"),
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


def test_progress_checkpoint_can_be_renewed_after_observed_progress() -> None:
    policy = TerminationPolicy(TerminationConfig(max_steps=3, max_seconds=100))
    state = SessionState(
        task="finish a multi-step task",
        step_count=3,
        step_limit=3,
        progress_revision=1,
        started_at=10,
    )

    assert policy.check(state, now=11) == "maximum agent step checkpoint reached (3)"
    assert state.made_progress_since_checkpoint

    state.renew_step_checkpoint(policy.config.max_steps)
    assert state.step_limit == 6
    assert not state.made_progress_since_checkpoint
    assert policy.check(state, now=11) is None


def test_optional_tool_call_limit_is_independent_from_model_steps() -> None:
    state = SessionState(task="bounded child", tool_call_count=10, initial_tool_call_count=2)
    policy = TerminationPolicy(TerminationConfig(max_tool_calls=8))

    assert policy.check(state) == "maximum tool calls reached (8)"
