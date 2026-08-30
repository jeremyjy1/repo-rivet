"""Pure reducer for run, model-call, and action lifecycle events."""

from __future__ import annotations

from datetime import UTC, datetime

from repo_rivet.actions.models import (
    ActionRecord,
    ActionResultSnapshot,
    ActionStatus,
    RecoveryState,
    RetryClass,
)
from repo_rivet.agent.phases import (
    DecisionEpoch,
    ModelCallRecord,
    ModelCallStatus,
    RunStatus,
    WaitKind,
    WaitState,
    WorkflowPhase,
)
from repo_rivet.agent.runtime import AgentRuntimeState
from repo_rivet.events.models import DomainEvent, DomainEventKind


class TransitionError(RuntimeError):
    """Raised when an event requests an impossible state transition."""


_ACTION_TRANSITIONS: dict[ActionStatus, set[ActionStatus]] = {
    ActionStatus.PROPOSED: {ActionStatus.PREPARED, ActionStatus.CANCELLED},
    ActionStatus.PREPARED: {
        ActionStatus.WAITING_APPROVAL,
        ActionStatus.APPROVED,
        ActionStatus.DISPATCHED,
        ActionStatus.OBSERVED,
        ActionStatus.CANCELLED,
    },
    ActionStatus.WAITING_APPROVAL: {
        ActionStatus.APPROVED,
        ActionStatus.OBSERVED,
        ActionStatus.CANCELLED,
    },
    ActionStatus.APPROVED: {
        ActionStatus.DISPATCHED,
        ActionStatus.OBSERVED,
        ActionStatus.CANCELLED,
    },
    ActionStatus.DISPATCHED: {ActionStatus.RUNNING, ActionStatus.OBSERVED},
    ActionStatus.RUNNING: {ActionStatus.OBSERVED, ActionStatus.INTERRUPTED_UNKNOWN},
    ActionStatus.OBSERVED: {ActionStatus.PREPARED, ActionStatus.APPLIED},
    ActionStatus.APPLIED: {ActionStatus.SUCCEEDED, ActionStatus.FAILED},
}


def _transition_action(action: ActionRecord, target: ActionStatus) -> None:
    if target not in _ACTION_TRANSITIONS.get(action.status, set()):
        raise TransitionError(
            f"Action {action.action_id} cannot transition from {action.status} to {target}"
        )
    action.status = target
    action.updated_at = datetime.now(UTC)


def _action(state: AgentRuntimeState, event: DomainEvent) -> ActionRecord:
    action_id = event.correlation_id or str(event.payload.get("action_id", ""))
    try:
        return state.actions[action_id]
    except KeyError:
        raise TransitionError(f"Unknown action for {event.kind}: {action_id}") from None


def reduce(state: AgentRuntimeState, event: DomainEvent) -> AgentRuntimeState:
    """Return a new state without performing any external side effect."""
    if event.seq != state.last_event_seq + 1:
        raise TransitionError(
            f"Event sequence mismatch: expected {state.last_event_seq + 1}, got {event.seq}"
        )
    value = state.model_copy(deep=True)
    value.state_version += 1
    value.last_event_seq = event.seq
    kind = event.kind

    if kind == DomainEventKind.RUN_ACTIVATED:
        value.status = RunStatus.ACTIVE
        value.phase = WorkflowPhase(event.payload.get("phase", WorkflowPhase.DECIDING))
        value.wait = None
        value.terminal_reason = None
    elif kind == DomainEventKind.RUNTIME_RECONCILED:
        if value.model_call is not None and value.model_call.status == ModelCallStatus.PENDING:
            value.model_call.status = ModelCallStatus.FAILED
        active = next((item for item in value.actions.values() if not item.terminal), None)
        if active is not None:
            if active.status == ActionStatus.OBSERVED and active.result is not None:
                value.phase = WorkflowPhase.APPLYING_OBSERVATION
            elif active.status in {ActionStatus.DISPATCHED, ActionStatus.RUNNING}:
                _transition_action(active, ActionStatus.INTERRUPTED_UNKNOWN)
                value.current_action_id = None
            else:
                _transition_action(active, ActionStatus.CANCELLED)
                # No external effect was dispatched, so the same proposal remains safe to
                # evaluate again in the resumed run.
                active.retryable = True
                value.current_action_id = None
        value.status = RunStatus.ACTIVE
        if active is None or active.status != ActionStatus.OBSERVED:
            value.phase = WorkflowPhase.RECOVERING
        value.wait = None
        recovery = event.payload.get("recovery")
        if recovery is not None:
            value.recovery = RecoveryState.model_validate(recovery)
    elif kind == DomainEventKind.PHASE_CHANGED:
        value.phase = WorkflowPhase(event.payload["phase"])
    elif kind == DomainEventKind.MODEL_CALL_STARTED:
        if value.model_call is not None and value.model_call.status == ModelCallStatus.PENDING:
            raise TransitionError("A model call is already pending")
        record = ModelCallRecord.model_validate(event.payload["model_call"])
        value.model_call = record
        value.decision_epoch = DecisionEpoch.model_validate(event.payload["decision_epoch"])
        value.status = RunStatus.WAITING
        value.wait = WaitState(
            kind=WaitKind.MODEL_RESPONSE,
            correlation_id=record.model_call_id,
            resume_phase=record.phase,
        )
    elif kind in {DomainEventKind.MODEL_CALL_FINISHED, DomainEventKind.MODEL_CALL_FAILED}:
        if value.model_call is None or value.model_call.status != ModelCallStatus.PENDING:
            raise TransitionError("No pending model call can receive this result")
        if event.correlation_id != value.model_call.model_call_id:
            raise TransitionError("Model result correlation does not match the pending call")
        value.model_call.status = (
            ModelCallStatus.COMPLETED
            if kind == DomainEventKind.MODEL_CALL_FINISHED
            else ModelCallStatus.FAILED
        )
        value.status = RunStatus.ACTIVE
        value.wait = None
    elif kind == DomainEventKind.ACTION_PROPOSED:
        record = ActionRecord.model_validate(event.payload["action"])
        active = [item.action_id for item in value.actions.values() if not item.terminal]
        if active:
            raise TransitionError(f"Cannot propose {record.action_id}; active action: {active[0]}")
        value.actions[record.action_id] = record
        value.current_action_id = record.action_id
        value.status = RunStatus.ACTIVE
        value.phase = WorkflowPhase.PREPARING_ACTION
    elif kind == DomainEventKind.ACTION_PREPARED:
        _transition_action(_action(value, event), ActionStatus.PREPARED)
    elif kind == DomainEventKind.APPROVAL_REQUESTED:
        action = _action(value, event)
        _transition_action(action, ActionStatus.WAITING_APPROVAL)
        value.status = RunStatus.WAITING
        value.wait = WaitState(
            kind=WaitKind.APPROVAL,
            correlation_id=event.correlation_id or action.action_id,
            resume_phase=WorkflowPhase.EXECUTING_ACTION,
            metadata=event.payload,
        )
    elif kind == DomainEventKind.APPROVAL_RESOLVED:
        action = _action(value, event)
        if bool(event.payload.get("approved")):
            _transition_action(action, ActionStatus.APPROVED)
        value.status = RunStatus.ACTIVE
        value.wait = None
    elif kind == DomainEventKind.ACTION_DISPATCHED:
        action = _action(value, event)
        if action.status == ActionStatus.PREPARED or action.status == ActionStatus.APPROVED:
            _transition_action(action, ActionStatus.DISPATCHED)
        else:
            raise TransitionError(f"Action {action.action_id} is not dispatchable")
        value.status = RunStatus.WAITING
        value.phase = WorkflowPhase.EXECUTING_ACTION
        value.wait = WaitState(
            kind=WaitKind.TOOL_COMPLETION,
            correlation_id=action.action_id,
            resume_phase=WorkflowPhase.APPLYING_OBSERVATION,
        )
    elif kind == DomainEventKind.ACTION_RUNNING:
        _transition_action(_action(value, event), ActionStatus.RUNNING)
    elif kind == DomainEventKind.ACTION_OBSERVED:
        action = _action(value, event)
        if action.status in {
            ActionStatus.PREPARED,
            ActionStatus.WAITING_APPROVAL,
            ActionStatus.APPROVED,
            ActionStatus.DISPATCHED,
            ActionStatus.RUNNING,
        }:
            _transition_action(action, ActionStatus.OBSERVED)
        else:
            raise TransitionError(f"Action {action.action_id} cannot accept an observation")
        action.result_event_id = event.event_id
        action.result = ActionResultSnapshot.model_validate(event.payload["result"])
        action.retryable = bool(event.payload.get("retryable", action.result.retryable))
        action.retry_class = RetryClass(event.payload.get("retry_class", RetryClass.NONE.value))
        value.pending_observation_ids.append(event.event_id)
        value.status = RunStatus.ACTIVE
        value.phase = WorkflowPhase.APPLYING_OBSERVATION
        value.wait = None
    elif kind == DomainEventKind.ACTION_RETRY_SCHEDULED:
        action = _action(value, event)
        result_event_id = action.result_event_id
        _transition_action(action, ActionStatus.PREPARED)
        action.attempt += 1
        action.result = None
        action.result_event_id = None
        action.retryable = False
        value.pending_observation_ids = [
            item for item in value.pending_observation_ids if item != result_event_id
        ]
        value.status = RunStatus.WAITING
        value.phase = WorkflowPhase.PREPARING_ACTION
        value.wait = WaitState(
            kind=WaitKind.RETRY_BACKOFF,
            correlation_id=action.action_id,
            resume_phase=WorkflowPhase.EXECUTING_ACTION,
            metadata={"attempt": action.attempt},
        )
    elif kind == DomainEventKind.OBSERVATION_APPLIED:
        action = _action(value, event)
        _transition_action(action, ActionStatus.APPLIED)
        action.result_applied = True
        semantic_key = event.payload.get("semantic_key")
        if isinstance(semantic_key, str) and semantic_key:
            action.semantic_key = semantic_key
        succeeded = bool(event.payload.get("succeeded"))
        _transition_action(
            action,
            ActionStatus.SUCCEEDED if succeeded else ActionStatus.FAILED,
        )
        value.pending_observation_ids = [
            item for item in value.pending_observation_ids if item != action.result_event_id
        ]
        value.current_action_id = None
        value.revisions.knowledge += int(bool(event.payload.get("new_knowledge", True)))
        if isinstance(event.payload.get("workspace_revision"), int):
            value.revisions.workspace = int(event.payload["workspace_revision"])
        value.status = RunStatus.ACTIVE
        value.phase = WorkflowPhase(event.payload.get("next_phase", WorkflowPhase.DECIDING))
    elif kind == DomainEventKind.OBSERVATION_DELIVERED:
        action = _action(value, event)
        if action.result_event_id is None:
            raise TransitionError("An action without an observation cannot be delivered")
        action.result_delivered_to_model = True
    elif kind == DomainEventKind.RECOVERY_ENTERED:
        value.recovery = RecoveryState.model_validate(event.payload["recovery"])
        value.status = RunStatus.ACTIVE
        value.phase = WorkflowPhase.RECOVERING
        value.wait = None
    elif kind == DomainEventKind.RECOVERY_CLEARED:
        value.recovery = None
        value.phase = WorkflowPhase(event.payload.get("phase", WorkflowPhase.DECIDING))
    elif kind == DomainEventKind.RUN_FINISHED:
        value.status = RunStatus(event.payload["status"])
        value.terminal_reason = event.payload.get("reason")
        value.wait = None
    else:  # pragma: no cover - exhaustive guard for future enum additions
        raise TransitionError(f"Unhandled domain event: {kind}")

    return value
