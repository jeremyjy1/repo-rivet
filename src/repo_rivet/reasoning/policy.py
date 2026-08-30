"""Controller-owned per-call reasoning effort selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from repo_rivet.llm.base import REASONING_EFFORTS, ReasoningEffort

ReasoningCallPhase = Literal[
    "skill_routing",
    "approval_review",
    "discovering",
    "planning",
    "acting",
    "editing",
    "recovering",
    "finalizing",
]

_EFFORT_INDEX = {effort: index for index, effort in enumerate(REASONING_EFFORTS)}
_PHASE_DEFAULTS: dict[ReasoningCallPhase, ReasoningEffort] = {
    "skill_routing": "low",
    "approval_review": "low",
    "discovering": "low",
    "planning": "medium",
    "acting": "low",
    "editing": "medium",
    "recovering": "high",
    "finalizing": "low",
}


class ReasoningPolicyMode(StrEnum):
    ADAPTIVE = "adaptive"
    FIXED = "fixed"


@dataclass(frozen=True, slots=True)
class ReasoningPolicySettings:
    mode: ReasoningPolicyMode = ReasoningPolicyMode.ADAPTIVE
    floor: ReasoningEffort = "low"
    ceiling: ReasoningEffort = "max"
    max_calls_per_run: int = 1
    xhigh_calls_per_run: int = 3


@dataclass(frozen=True, slots=True)
class ReasoningContext:
    phase: ReasoningCallPhase
    affected_file_count: int = 0
    unresolved_unknown_count: int = 0
    failed_hypothesis_count: int = 0
    cross_module_change: bool = False
    architectural_decision: bool = False
    conflicting_evidence: bool = False
    stale_snapshot_conflict: bool = False
    next_action_already_known: bool = False
    latency_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class ReasoningUsage:
    current_step: int = 0
    max_calls: int = 0
    xhigh_calls: int = 0


@dataclass(frozen=True, slots=True)
class ReasoningLease:
    effort: ReasoningEffort
    reason: str
    phase: ReasoningCallPhase
    issued_at_step: int
    valid_for_calls: int = 1


class AdaptiveReasoningPolicy:
    """Choose one bounded reasoning lease from deterministic controller state."""

    def choose(
        self,
        context: ReasoningContext,
        settings: ReasoningPolicySettings,
        usage: ReasoningUsage,
    ) -> ReasoningLease:
        if settings.mode == ReasoningPolicyMode.FIXED:
            return ReasoningLease(
                effort=clamp_effort(settings.ceiling, settings.floor, settings.ceiling),
                reason=f"fixed reasoning level selected by user: {settings.ceiling}",
                phase=context.phase,
                issued_at_step=usage.current_step,
            )

        effort = _PHASE_DEFAULTS[context.phase]
        reasons = [f"{context.phase} phase starts at {effort}"]
        if context.cross_module_change:
            effort = bump_effort(effort)
            reasons.append("change spans multiple modules")
        if context.unresolved_unknown_count >= 2:
            effort = bump_effort(effort)
            reasons.append("multiple important unknowns remain")
        if context.architectural_decision:
            effort = bump_effort(effort)
            reasons.append("an architectural trade-off is required")
        if context.conflicting_evidence:
            effort = bump_effort(effort)
            reasons.append("observations conflict")
        if context.failed_hypothesis_count >= 2:
            effort = bump_effort(effort)
            reasons.append("at least two evidence-backed attempts failed")
        if context.stale_snapshot_conflict:
            effort = bump_effort(effort, 2)
            reasons.append("a stale snapshot or edit conflict needs recovery")

        if context.next_action_already_known:
            effort = "low"
            reasons.append("the next approved action is already known")
        if context.latency_sensitive and effort_index(effort) > effort_index("medium"):
            effort = "medium"
            reasons.append("this phase is latency sensitive")

        if effort == "max" and usage.max_calls >= settings.max_calls_per_run:
            effort = "xhigh"
            reasons.append("the run-level Max lease budget is exhausted")
        if effort == "xhigh" and usage.xhigh_calls >= settings.xhigh_calls_per_run:
            effort = "high"
            reasons.append("the run-level XHigh lease budget is exhausted")

        bounded = clamp_effort(effort, settings.floor, settings.ceiling)
        if bounded != effort:
            reasons.append(f"bounded by the user range {settings.floor}..{settings.ceiling}")
        return ReasoningLease(
            effort=bounded,
            reason="; ".join(reasons),
            phase=context.phase,
            issued_at_step=usage.current_step,
        )


def effort_index(effort: ReasoningEffort) -> int:
    return _EFFORT_INDEX[effort]


def bump_effort(effort: ReasoningEffort, levels: int = 1) -> ReasoningEffort:
    index = min(len(REASONING_EFFORTS) - 1, effort_index(effort) + levels)
    return REASONING_EFFORTS[index]


def clamp_effort(
    effort: ReasoningEffort,
    floor: ReasoningEffort,
    ceiling: ReasoningEffort,
) -> ReasoningEffort:
    floor_index = effort_index(floor)
    ceiling_index = effort_index(ceiling)
    selected = max(floor_index, min(effort_index(effort), ceiling_index))
    return REASONING_EFFORTS[selected]


def map_to_supported_effort(
    desired: ReasoningEffort,
    supported: tuple[ReasoningEffort, ...],
) -> ReasoningEffort:
    """Select the strongest provider tier that does not exceed the desired effort."""
    candidates = [item for item in supported if effort_index(item) <= effort_index(desired)]
    if candidates:
        return max(candidates, key=effort_index)
    return min(supported, key=effort_index)
