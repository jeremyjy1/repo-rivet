from __future__ import annotations

from repo_rivet.actions.identity import ActionIdentity
from repo_rivet.actions.models import (
    ActionResultSnapshot,
    ActionStatus,
    DuplicateDisposition,
)
from repo_rivet.actions.registry import ActionRegistry
from repo_rivet.agent.phases import RevisionVector, WorkflowPhase
from repo_rivet.agent.runtime_state import AgentRuntimeState
from repo_rivet.memory.models import MemoryState
from repo_rivet.tools.base import ToolCall


def test_verification_result_is_reused_only_at_same_revision() -> None:
    memory = MemoryState(session_id="session")
    runtime = AgentRuntimeState.create("session", workspace_revision=3)
    call = ToolCall(
        id="verify-1",
        name="run_verification",
        arguments={"check_id": "tests"},
    )
    key = ActionIdentity.build(
        call,
        context=memory,
        revisions=runtime.revisions,
        plan_step_id="verify",
    )
    record = ActionRegistry.build_record(
        call,
        semantic_key=key,
        runtime=runtime,
        revisions=runtime.revisions,
        plan_step_id="verify",
        continuation_phase=WorkflowPhase.VERIFYING,
    )
    record.status = ActionStatus.SUCCEEDED
    record.result = ActionResultSnapshot(ok=True, output="passed")
    record.result_applied = True
    record.result_delivered_to_model = True
    runtime.actions[record.action_id] = record

    same = ActionRegistry().classify(
        call,
        runtime=runtime,
        context=memory,
        plan_step_id="verify",
    )
    assert same.disposition == DuplicateDisposition.REUSE_RESULT

    runtime.revisions = RevisionVector(workspace=4)
    changed = ActionRegistry().classify(
        call,
        runtime=runtime,
        context=memory,
        plan_step_id="verify",
    )
    assert changed.disposition == DuplicateDisposition.EXECUTE_NEW


def test_failed_action_requires_an_alternative_not_external_replay() -> None:
    memory = MemoryState(session_id="session")
    runtime = AgentRuntimeState.create("session")
    call = ToolCall(id="command-1", name="run_command", arguments={"command": "false"})
    key = ActionIdentity.build(
        call,
        context=memory,
        revisions=runtime.revisions,
        plan_step_id=None,
    )
    record = ActionRegistry.build_record(
        call,
        semantic_key=key,
        runtime=runtime,
        revisions=runtime.revisions,
        plan_step_id=None,
        continuation_phase=WorkflowPhase.RECOVERING,
    )
    record.status = ActionStatus.FAILED
    record.result = ActionResultSnapshot(ok=False, output="", error="failed")
    record.result_applied = True
    record.result_delivered_to_model = True
    runtime.actions[record.action_id] = record

    classification = ActionRegistry().classify(
        call,
        runtime=runtime,
        context=memory,
        plan_step_id=None,
    )

    assert classification.disposition == DuplicateDisposition.REQUIRE_ALTERNATIVE
    assert classification.previous is record
