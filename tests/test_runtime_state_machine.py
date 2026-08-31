from __future__ import annotations

import pytest

from repo_rivet.actions.models import ActionRecord, ActionResultSnapshot, ActionStatus
from repo_rivet.agent.controller import AgentController
from repo_rivet.agent.invariants import StateInvariantError, assert_state_invariants
from repo_rivet.agent.phases import (
    DecisionEpoch,
    ModelCallRecord,
    RunStatus,
    WaitKind,
    WorkflowPhase,
)
from repo_rivet.agent.runtime import AgentRuntimeState
from repo_rivet.agent.runtime_kernel import RuntimeKernel
from repo_rivet.events.models import DomainEventKind
from repo_rivet.llm.base import ModelResponse
from repo_rivet.memory.models import MemoryState, Message
from tests.fakes import FakeModelClient, FakeToolRegistry


def epoch() -> DecisionEpoch:
    return DecisionEpoch(
        phase=WorkflowPhase.DECIDING,
        workspace_revision=0,
        knowledge_revision=0,
        plan_revision=0,
        verification_plan_revision=0,
        verification_state_hash="verification",
    )


def action(runtime: AgentRuntimeState) -> ActionRecord:
    return ActionRecord(
        action_id="action-1",
        semantic_key="read:key",
        tool_call_id="call-1",
        tool_name="read_file",
        normalized_arguments={"path": "app.py"},
        revisions=runtime.revisions,
    )


def test_model_call_is_an_explicit_wait_and_rejects_concurrent_action() -> None:
    kernel = RuntimeKernel(AgentRuntimeState.create("session"))
    kernel.dispatch(
        DomainEventKind.RUN_ACTIVATED,
        payload={"phase": WorkflowPhase.DECIDING.value},
    )
    decision_epoch = epoch()
    model_call = ModelCallRecord(
        model_call_id="model-1",
        decision_epoch_id=decision_epoch.epoch_id,
        phase=WorkflowPhase.DECIDING,
    )
    kernel.dispatch(
        DomainEventKind.MODEL_CALL_STARTED,
        correlation_id=model_call.model_call_id,
        payload={
            "model_call": model_call.model_dump(mode="json"),
            "decision_epoch": decision_epoch.model_dump(mode="json"),
        },
    )

    assert kernel.state.status == RunStatus.WAITING
    assert kernel.state.wait is not None
    assert kernel.state.wait.kind == WaitKind.MODEL_RESPONSE
    with pytest.raises(StateInvariantError):
        kernel.dispatch(
            DomainEventKind.ACTION_PROPOSED,
            correlation_id="action-1",
            payload={"action": action(kernel.state).model_dump(mode="json")},
        )

    kernel.dispatch(
        DomainEventKind.MODEL_CALL_FINISHED,
        correlation_id=model_call.model_call_id,
    )
    assert kernel.state.status == RunStatus.ACTIVE
    assert kernel.state.wait is None


def test_action_lifecycle_persists_approval_execution_and_observation() -> None:
    kernel = RuntimeKernel(AgentRuntimeState.create("session"))
    kernel.dispatch(
        DomainEventKind.RUN_ACTIVATED,
        payload={"phase": WorkflowPhase.DECIDING.value},
    )
    record = action(kernel.state)
    kernel.dispatch(
        DomainEventKind.ACTION_PROPOSED,
        correlation_id=record.action_id,
        payload={"action": record.model_dump(mode="json")},
    )
    kernel.dispatch(DomainEventKind.ACTION_PREPARED, correlation_id=record.action_id)
    kernel.dispatch(DomainEventKind.APPROVAL_REQUESTED, correlation_id=record.action_id)
    assert kernel.state.wait is not None
    assert kernel.state.wait.kind == WaitKind.APPROVAL
    kernel.dispatch(
        DomainEventKind.APPROVAL_RESOLVED,
        correlation_id=record.action_id,
        payload={"approved": True},
    )
    kernel.dispatch(DomainEventKind.ACTION_DISPATCHED, correlation_id=record.action_id)
    kernel.dispatch(DomainEventKind.ACTION_RUNNING, correlation_id=record.action_id)
    kernel.dispatch(
        DomainEventKind.ACTION_OBSERVED,
        correlation_id=record.action_id,
        payload={"result": ActionResultSnapshot(ok=True, output="content").model_dump(mode="json")},
    )
    kernel.dispatch(
        DomainEventKind.OBSERVATION_APPLIED,
        correlation_id=record.action_id,
        payload={"succeeded": True, "workspace_revision": 0},
    )
    kernel.dispatch(DomainEventKind.OBSERVATION_DELIVERED, correlation_id=record.action_id)

    completed = kernel.state.actions[record.action_id]
    assert completed.status == ActionStatus.SUCCEEDED
    assert completed.result_applied
    assert completed.result_delivered_to_model
    assert kernel.state.current_action_id is None
    assert_state_invariants(kernel.state)


def test_delegation_dispatch_uses_subagent_result_wait_state() -> None:
    kernel = RuntimeKernel(AgentRuntimeState.create("session"))
    kernel.dispatch(
        DomainEventKind.RUN_ACTIVATED,
        payload={"phase": WorkflowPhase.DECIDING.value},
    )
    record = action(kernel.state)
    kernel.dispatch(
        DomainEventKind.ACTION_PROPOSED,
        correlation_id=record.action_id,
        payload={"action": record.model_dump(mode="json")},
    )
    kernel.dispatch(DomainEventKind.ACTION_PREPARED, correlation_id=record.action_id)
    kernel.dispatch(
        DomainEventKind.ACTION_DISPATCHED,
        correlation_id=record.action_id,
        payload={"wait_kind": WaitKind.SUBAGENT_RESULTS.value},
    )

    assert kernel.state.status == RunStatus.WAITING
    assert kernel.state.wait is not None
    assert kernel.state.wait.kind == WaitKind.SUBAGENT_RESULTS
    assert_state_invariants(kernel.state)


def test_restart_reconciliation_never_blindly_reexecutes_running_action() -> None:
    kernel = RuntimeKernel(AgentRuntimeState.create("session"))
    kernel.dispatch(
        DomainEventKind.RUN_ACTIVATED,
        payload={"phase": WorkflowPhase.DECIDING.value},
    )
    record = action(kernel.state)
    kernel.dispatch(
        DomainEventKind.ACTION_PROPOSED,
        correlation_id=record.action_id,
        payload={"action": record.model_dump(mode="json")},
    )
    kernel.dispatch(DomainEventKind.ACTION_PREPARED, correlation_id=record.action_id)
    kernel.dispatch(DomainEventKind.ACTION_DISPATCHED, correlation_id=record.action_id)
    kernel.dispatch(DomainEventKind.ACTION_RUNNING, correlation_id=record.action_id)

    kernel.dispatch(DomainEventKind.RUNTIME_RECONCILED)

    reconciled = kernel.state.actions[record.action_id]
    assert reconciled.status == ActionStatus.INTERRUPTED_UNKNOWN
    assert kernel.state.phase == WorkflowPhase.RECOVERING
    assert kernel.state.current_action_id is None
    assert kernel.state.wait is None


def test_observed_action_is_applied_after_restart_without_tool_reexecution() -> None:
    kernel = RuntimeKernel(AgentRuntimeState.create("session"))
    kernel.dispatch(
        DomainEventKind.RUN_ACTIVATED,
        payload={"phase": WorkflowPhase.DECIDING.value},
    )
    record = action(kernel.state)
    kernel.dispatch(
        DomainEventKind.ACTION_PROPOSED,
        correlation_id=record.action_id,
        payload={"action": record.model_dump(mode="json")},
    )
    kernel.dispatch(DomainEventKind.ACTION_PREPARED, correlation_id=record.action_id)
    kernel.dispatch(DomainEventKind.ACTION_DISPATCHED, correlation_id=record.action_id)
    kernel.dispatch(DomainEventKind.ACTION_RUNNING, correlation_id=record.action_id)
    kernel.dispatch(
        DomainEventKind.ACTION_OBSERVED,
        correlation_id=record.action_id,
        payload={"result": ActionResultSnapshot(ok=True, output="content").model_dump(mode="json")},
    )
    memory = MemoryState(session_id="session", runtime=kernel.state)
    memory.messages.append(
        Message(
            role="assistant",
            tool_calls=[
                {
                    "id": record.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": record.tool_name,
                        "arguments": '{"path":"app.py"}',
                    },
                }
            ],
        )
    )
    tools = FakeToolRegistry([])

    result = AgentController(
        model_client=FakeModelClient([ModelResponse(content="Recovered observation applied.")]),
        tool_registry=tools,
    ).run("continue", memory=memory)

    assert result.status == "success"
    assert tools.calls == []
    assert record.action_id in memory.applied_action_ids
    assert memory.runtime is not None
    recovered = memory.runtime.actions[record.action_id]
    assert recovered.status == ActionStatus.SUCCEEDED
    assert recovered.result_delivered_to_model
