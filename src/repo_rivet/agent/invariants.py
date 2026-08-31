"""Fail-fast invariants for the runtime state."""

from repo_rivet.actions.models import ActionStatus
from repo_rivet.agent.phases import MODEL_PHASES, ModelCallStatus, RunStatus, WaitKind
from repo_rivet.agent.runtime import AgentRuntimeState


class StateInvariantError(RuntimeError):
    """The Controller produced a state that cannot occur in a valid run."""


def assert_state_invariants(state: AgentRuntimeState) -> None:
    active_actions = [action for action in state.actions.values() if not action.terminal]
    if len(active_actions) > 1:
        raise StateInvariantError("A run cannot have more than one active primary action")
    if state.current_action_id is None and active_actions:
        raise StateInvariantError("An active action exists without current_action_id")
    if state.current_action_id is not None:
        current = state.actions.get(state.current_action_id)
        if current is None or current.terminal:
            raise StateInvariantError("current_action_id does not name an active action")
    if state.status == RunStatus.WAITING and state.wait is None:
        raise StateInvariantError("WAITING requires a WaitState")
    if state.status != RunStatus.WAITING and state.wait is not None:
        raise StateInvariantError("Only WAITING may retain a WaitState")
    pending_model = (
        state.model_call is not None and state.model_call.status == ModelCallStatus.PENDING
    )
    if pending_model and (state.wait is None or state.wait.kind != WaitKind.MODEL_RESPONSE):
        raise StateInvariantError("A pending model call requires MODEL_RESPONSE wait state")
    if state.wait is not None and state.wait.kind == WaitKind.MODEL_RESPONSE and not pending_model:
        raise StateInvariantError("MODEL_RESPONSE wait has no pending model call")
    if (
        state.wait is not None
        and state.wait.kind
        in {
            WaitKind.APPROVAL,
            WaitKind.TOOL_COMPLETION,
            WaitKind.SUBAGENT_RESULTS,
        }
        and state.phase in MODEL_PHASES
    ):
        raise StateInvariantError("The model cannot be called while an external effect is pending")
    for action in state.actions.values():
        if (
            action.status in {ActionStatus.SUCCEEDED, ActionStatus.FAILED}
            and not action.result_applied
        ):
            raise StateInvariantError("A terminal observed action must have an applied result")
