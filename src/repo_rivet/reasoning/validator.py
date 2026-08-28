"""Consistency rules between declared decisions and actual model tool calls."""

from repo_rivet.reasoning.models import ReasoningEvent, ReasoningPhase
from repo_rivet.tools.base import ToolCall


class DecisionValidationError(ValueError):
    """A model turn cannot safely execute its declared actions."""


def validate_decision_for_actions(
    decision: ReasoningEvent | None,
    action_calls: list[ToolCall],
    *,
    mutating_calls: list[ToolCall],
    require_decision: bool,
) -> None:
    """Require one matching decision before any state-changing operation."""
    if len(mutating_calls) > 1:
        raise DecisionValidationError(
            "At most one state-changing tool may run in a model turn; split the operations."
        )
    if mutating_calls and require_decision:
        if decision is None or decision.phase != ReasoningPhase.DECISION:
            raise DecisionValidationError(
                "A decision record is required for this state-changing tool. Include "
                "record_decision and the tool in the same model response, or record the "
                "decision alone and issue its matching tool in the immediately following "
                "model response."
            )
        if decision.next_action is None:
            raise DecisionValidationError(
                "The decision must declare next_tool and expected_result before execution."
            )

    if decision is None or decision.next_action is None or not action_calls:
        return
    declared_targets = mutating_calls or action_calls
    actual_tools = {call.name for call in declared_targets}
    if decision.next_action.tool_name not in actual_tools:
        raise DecisionValidationError(
            f"Decision declared {decision.next_action.tool_name}, but the actual tools were "
            f"{', '.join(sorted(actual_tools))}."
        )
