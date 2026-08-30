from repo_rivet.reasoning.policy import (
    AdaptiveReasoningPolicy,
    ReasoningContext,
    ReasoningPolicyMode,
    ReasoningPolicySettings,
    ReasoningUsage,
    map_to_supported_effort,
)


def test_adaptive_policy_selects_each_call_from_phase_and_facts() -> None:
    policy = AdaptiveReasoningPolicy()
    settings = ReasoningPolicySettings(ceiling="max")

    discovering = policy.choose(ReasoningContext(phase="discovering"), settings, ReasoningUsage())
    planning = policy.choose(
        ReasoningContext(
            phase="planning",
            cross_module_change=True,
            architectural_decision=True,
        ),
        settings,
        ReasoningUsage(current_step=2),
    )
    known_action = policy.choose(
        ReasoningContext(
            phase="recovering",
            conflicting_evidence=True,
            stale_snapshot_conflict=True,
            next_action_already_known=True,
        ),
        settings,
        ReasoningUsage(current_step=3),
    )

    assert discovering.effort == "low"
    assert planning.effort == "xhigh"
    assert known_action.effort == "max"
    assert "stale snapshot or edit conflict" in known_action.reason
    assert "next approved action is already known" not in known_action.reason
    assert known_action.valid_for_calls == 1


def test_adaptive_policy_limits_max_and_xhigh_leases_per_run() -> None:
    policy = AdaptiveReasoningPolicy()
    context = ReasoningContext(
        phase="recovering",
        cross_module_change=True,
        conflicting_evidence=True,
    )
    settings = ReasoningPolicySettings(
        ceiling="max",
        max_calls_per_run=1,
        xhigh_calls_per_run=1,
    )

    first = policy.choose(context, settings, ReasoningUsage())
    second = policy.choose(context, settings, ReasoningUsage(max_calls=1))
    third = policy.choose(
        context,
        settings,
        ReasoningUsage(max_calls=1, xhigh_calls=1),
    )

    assert first.effort == "max"
    assert second.effort == "xhigh"
    assert third.effort == "high"


def test_fixed_policy_uses_selected_level_for_every_call() -> None:
    lease = AdaptiveReasoningPolicy().choose(
        ReasoningContext(phase="finalizing", latency_sensitive=True),
        ReasoningPolicySettings(
            mode=ReasoningPolicyMode.FIXED,
            floor="low",
            ceiling="max",
            max_calls_per_run=0,
        ),
        ReasoningUsage(max_calls=99),
    )

    assert lease.effort == "max"
    assert "fixed reasoning level" in lease.reason


def test_provider_mapping_never_exceeds_requested_effort() -> None:
    supported = ("low", "high", "max")

    assert map_to_supported_effort("medium", supported) == "low"
    assert map_to_supported_effort("xhigh", supported) == "high"
    assert map_to_supported_effort("max", supported) == "max"
