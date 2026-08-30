from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from repo_rivet.agent.controller import AgentController
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy
from repo_rivet.config import ApiConfig
from repo_rivet.editing.document import TextDocument
from repo_rivet.llm.base import ModelResponse
from repo_rivet.memory.models import MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.planning.classifier import (
    OpenAIPlanClassifier,
    PlanClassification,
    WorkspacePlanningSummary,
    summarize_workspace,
)
from repo_rivet.planning.errors import PlanModeViolation
from repo_rivet.planning.models import (
    PlanDraft,
    PlanStatus,
    PlanStepStatus,
    WorkflowMode,
)
from repo_rivet.planning.policy import AutoPlanMode, AutoPlanPolicy
from repo_rivet.planning.runtime import PLANNING_TOOL_NAMES, PlanRuntime
from repo_rivet.reasoning.models import ObservationEvent
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.tools.planning import SubmitPlanTool
from repo_rivet.tools.registry import create_default_registry
from tests.fakes import FakeModelClient


def observation(session_id: str = "plan-test") -> ObservationEvent:
    return ObservationEvent(
        event_id="obs-read",
        session_id=session_id,
        step=1,
        tool_call_id="read-1",
        tool_name="read_file",
        ok=True,
        result_summary="Read app.py:1-2.",
        affected_paths=["app.py"],
    )


def plan_payload(*, operation: str = "edit", target: str = "app.py") -> dict[str, object]:
    return {
        "goal": "Make the requested bounded change",
        "constraints": ["Keep behavior outside the target unchanged"],
        "assumptions": ["The current implementation is covered by the check"],
        "evidence_refs": ["obs-read"],
        "steps": [
            {
                "step_id": "change",
                "title": "Apply the change",
                "intent": "Update only the inspected target",
                "evidence_refs": ["obs-read"],
                "operation": operation,
                "target_files": [target],
                "verification_ids": ["tests"],
                "risk": "low",
            },
            {
                "step_id": "verify",
                "title": "Verify the result",
                "intent": "Run the registered focused check",
                "evidence_refs": ["obs-read"],
                "operation": "verify",
                "verification_ids": ["tests"],
                "depends_on": ["change"],
                "risk": "low",
            },
        ],
        "verification": [
            {
                "check_id": "tests",
                "title": "Focused tests",
                "success_criteria": "The registered check passes",
            }
        ],
        "risks": ["A boundary case may need an additional test"],
    }


def inspected_memory(workspace: Path) -> MemoryState:
    path = workspace / "app.py"
    path.write_text("value = 1\n", encoding="utf-8")
    snapshot = TextDocument.load(path).to_snapshot(relative_path="app.py")
    memory = MemoryState(session_id="plan-test")
    memory.observation_events.append(observation())
    memory.current_snapshots["app.py"] = snapshot.snapshot_id
    return memory


def test_plan_draft_rejects_cycles_and_missing_verify_steps() -> None:
    payload = plan_payload()
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps[0]["depends_on"] = ["verify"]  # type: ignore[index]

    with pytest.raises(ValueError, match="depends on later steps"):
        PlanDraft.model_validate(payload)

    payload = plan_payload()
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps.pop()
    with pytest.raises(ValueError, match="lack executable verify steps"):
        PlanDraft.model_validate(payload)


def test_plan_runtime_executes_snapshot_bound_directory_delete_step(tmp_path: Path) -> None:
    directory = tmp_path / "generated"
    directory.mkdir()
    (directory / "artifact.txt").write_text("generated", encoding="utf-8")
    memory = MemoryState(session_id="plan-test")
    listed = observation().model_copy(
        update={
            "tool_name": "list_files",
            "result_summary": "Listed generated.",
            "affected_paths": ["generated"],
        }
    )
    memory.observation_events.append(listed)
    runtime = PlanRuntime(WorkspacePathPolicy(tmp_path))
    runtime.bind(memory)
    artifact = runtime.submit({"plan": plan_payload(operation="delete", target="generated")})
    runtime.approve()
    call = ToolCall(
        id="delete-1",
        name="delete_path",
        arguments={"path": "generated", "recursive": True},
    )

    assert artifact.snapshots["generated"]
    assert runtime.validate_action(call) is None
    runtime.start_action()
    memory.workspace_revision = 1
    runtime.observe_action(
        call,
        ToolResult(
            ok=True,
            output="Deleted directory generated",
            metadata={"path": "generated", "workspace_revision": 1, "deleted": True},
        ),
        evidence_ref="obs-delete",
    )

    assert artifact.steps[0].status == PlanStepStatus.COMPLETED
    assert "generated" not in artifact.execution_snapshots


def test_approved_delete_step_uses_plan_as_decision_record(tmp_path: Path) -> None:
    memory = inspected_memory(tmp_path)
    registry = create_default_registry(tmp_path)
    registry.plan_runtime.bind(memory)
    artifact = registry.plan_runtime.submit(
        {"plan": plan_payload(operation="delete", target="app.py")}
    )
    registry.plan_runtime.approve()
    registry.verification_runtime.bind(memory)
    registry.verification_runtime.register_plan(
        {
            "requirements": ["tests"],
            "checks": [
                {
                    "check_id": "tests",
                    "title": "Confirm workspace remains valid",
                    "kind": "test",
                    "command": {"program": "true", "args": [], "cwd": "."},
                    "criteria": {"expected_exit_codes": [0]},
                    "required": True,
                    "provenance": "model",
                }
            ],
        }
    )
    delete = ToolCall(
        id="delete-with-plan",
        name="delete_path",
        arguments={"path": "app.py"},
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[delete]),
            ModelResponse(content="Deleted the planned file and verified the workspace."),
        ]
    )
    events: list[tuple[str, dict[str, object]]] = []

    class RecordingSink:
        def log(self, event_type: str, **data: object) -> None:
            events.append((event_type, data))

    result = AgentController(
        model_client=model,
        tool_registry=registry,
        event_logger=RecordingSink(),
    ).run("Execute the approved deletion plan", memory=memory)

    assert result.status == "success"
    assert not (tmp_path / "app.py").exists()
    assert artifact.status == PlanStatus.COMPLETED
    assert not any(event.phase.value == "decision" for event in memory.reasoning_events)
    authorized = next(data for name, data in events if name == "plan_authorized_action")
    assert authorized["tool"] == "delete_path"
    assert authorized["plan_id"] == artifact.plan_id
    assert authorized["plan_step_id"] == "change"
    assert authorized["authority"] == "approved_implementation_plan"


def test_plan_tool_schema_does_not_expose_controller_owned_progress() -> None:
    schema = SubmitPlanTool().schema
    serialized = str(schema)

    assert "last_observation_ref" not in serialized
    assert "last_error" not in serialized
    assert "PlanStepStatus" not in serialized


def test_plan_runtime_binds_snapshot_and_rejects_stale_execution(tmp_path: Path) -> None:
    memory = inspected_memory(tmp_path)
    runtime = PlanRuntime(WorkspacePathPolicy(tmp_path))
    runtime.bind(memory)

    artifact = runtime.submit({"plan": plan_payload()})

    assert artifact.status == PlanStatus.READY
    assert artifact.workspace_revision == 0
    assert set(artifact.snapshots) == {"app.py"}
    assert memory.workflow_mode == WorkflowMode.PLAN_READY

    store = MemoryStore(tmp_path / "session")
    store.save_state(memory, status="plan_ready")
    restored = store.load_state()
    assert restored.plan_artifact == artifact
    assert restored.workflow_mode == WorkflowMode.PLAN_READY

    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Plan is stale"):
        runtime.approve()
    assert artifact.status == PlanStatus.STALE
    assert memory.workflow_mode == WorkflowMode.PLAN_READY


def test_plan_runtime_rejects_duplicate_normalized_create_targets(tmp_path: Path) -> None:
    memory = MemoryState(session_id="duplicate-create")
    memory.observation_events.append(observation("duplicate-create"))
    runtime = PlanRuntime(WorkspacePathPolicy(tmp_path))
    runtime.bind(memory)
    payload = plan_payload(operation="create", target="new.py")
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps.insert(
        1,
        {
            "step_id": "create-again",
            "title": "Create the same file again",
            "intent": "This impossible duplicate must be rejected",
            "evidence_refs": ["obs-read"],
            "operation": "create",
            "target_files": ["./new.py"],
            "verification_ids": ["tests"],
            "depends_on": ["change"],
            "risk": "low",
        },
    )
    steps[2]["depends_on"] = ["create-again"]  # type: ignore[index]

    with pytest.raises(ValueError, match="create target new.py is repeated"):
        runtime.submit({"plan": payload})


def test_plan_runtime_advances_only_from_typed_success_observations(tmp_path: Path) -> None:
    memory = inspected_memory(tmp_path)
    runtime = PlanRuntime(WorkspacePathPolicy(tmp_path))
    runtime.bind(memory)
    artifact = runtime.submit({"plan": plan_payload()})
    runtime.approve()

    wrong = ToolCall(id="command", name="run_command", arguments={"command": "true"})
    assert "requires edit" in (runtime.validate_action(wrong) or "")

    edit = ToolCall(id="edit", name="edit_file", arguments={"path": "app.py"})
    assert runtime.validate_action(edit) is None
    runtime.start_action()
    runtime.observe_action(
        edit,
        ToolResult(
            ok=False,
            output="",
            error="denied by user",
            error_code="approval_denied",
        ),
        evidence_ref="obs-edit-blocked",
    )
    assert artifact.steps[0].status == PlanStepStatus.BLOCKED
    assert artifact.current_step is artifact.steps[0]

    runtime.start_action()
    runtime.observe_action(
        edit,
        ToolResult(ok=True, output="missing typed edit result"),
        evidence_ref="obs-edit-failed",
    )
    assert artifact.steps[0].status == PlanStepStatus.FAILED

    runtime.start_action()
    (tmp_path / "app.py").write_text("value = 2\n", encoding="utf-8")
    new_snapshot = TextDocument.load(tmp_path / "app.py").to_snapshot(relative_path="app.py")
    runtime.observe_action(
        edit,
        ToolResult(
            ok=True,
            output="edited",
            metadata={
                "path": "app.py",
                "workspace_revision": 1,
                "new_snapshot_id": new_snapshot.snapshot_id,
            },
        ),
        evidence_ref="obs-edit",
    )
    memory.workspace_revision = 1
    assert artifact.steps[0].status == PlanStepStatus.COMPLETED
    assert artifact.current_step is artifact.steps[1]
    assert runtime.stale_reasons() == []

    verify = ToolCall(
        id="verify",
        name="run_verification",
        arguments={"check_id": "tests"},
    )
    runtime.start_action()
    runtime.observe_action(
        verify,
        ToolResult(ok=True, output="untyped verification"),
        evidence_ref="obs-verify-failed",
    )
    assert artifact.steps[1].status == PlanStepStatus.FAILED

    runtime.start_action()
    runtime.observe_action(
        verify,
        ToolResult(
            ok=True,
            output="passed",
            metadata={"verification_result": {"status": "passed"}},
        ),
        evidence_ref="obs-verify",
    )
    assert artifact.status == PlanStatus.COMPLETED
    assert artifact.current_step is None


def test_registered_check_satisfies_command_step_and_duplicate_verify_step(
    tmp_path: Path,
) -> None:
    memory = inspected_memory(tmp_path)
    runtime = PlanRuntime(WorkspacePathPolicy(tmp_path))
    runtime.bind(memory)
    payload = plan_payload(operation="command")
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps[0]["target_files"] = []  # type: ignore[index]
    artifact = runtime.submit({"plan": payload})
    runtime.approve()
    check = ToolCall(
        id="build",
        name="run_verification",
        arguments={"check_id": "tests"},
    )

    assert runtime.validate_action(check) is None
    runtime.start_action()
    runtime.observe_action(
        check,
        ToolResult(
            ok=True,
            output="passed",
            metadata={
                "exit_code": 0,
                "verification_result": {"status": "passed"},
            },
        ),
        evidence_ref="obs-build",
    )

    assert [step.status for step in artifact.steps] == [
        PlanStepStatus.COMPLETED,
        PlanStepStatus.COMPLETED,
    ]
    assert artifact.status == PlanStatus.COMPLETED
    assert artifact.current_step is None


def test_controller_normalizes_exact_verify_command_to_registered_check(tmp_path: Path) -> None:
    memory = inspected_memory(tmp_path)
    registry = create_default_registry(tmp_path)
    registry.plan_runtime.bind(memory)
    payload = plan_payload(operation="command")
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps[0]["target_files"] = []  # type: ignore[index]
    artifact = registry.plan_runtime.submit({"plan": payload})
    registry.plan_runtime.approve()
    registry.verification_runtime.bind(memory)
    registry.verification_runtime.register_plan(
        {
            "requirements": ["tests"],
            "checks": [
                {
                    "check_id": "tests",
                    "title": "Exact registered command",
                    "kind": "test",
                    "command": {"program": "true", "args": [], "cwd": "."},
                    "criteria": {"expected_exit_codes": [0]},
                    "required": True,
                    "provenance": "model",
                }
            ],
        }
    )
    requested = ToolCall(
        id="verify-as-command",
        name="run_command",
        arguments={"command": "true", "cwd": ".", "timeout_seconds": 60},
    )
    prior_decision = ToolCall(
        id="decision-as-command",
        name="record_decision",
        arguments={
            "phase": "decision",
            "current_goal": "Execute the approved command step",
            "summary": "Run the exact registered command for the current plan step.",
            "next_tool": "run_command",
            "next_tool_argument_summary": "Run true in the workspace",
            "expected_result": "The registered check passes",
            "confidence": 0.9,
        },
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[prior_decision]),
            ModelResponse(tool_calls=[requested]),
            ModelResponse(content="The registered check passed."),
        ]
    )

    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def log(self, event_type: str, **data: object) -> None:
            self.events.append((event_type, data))

    events = RecordingSink()
    result = AgentController(
        model_client=model,
        tool_registry=registry,
        event_logger=events,
    ).run("Execute the approved plan", memory=memory)

    assert result.status == "success"
    assert artifact.status == PlanStatus.COMPLETED
    assert memory.verification_results["tests"].status.value == "passed"
    normalized = next(data for name, data in events.events if name == "plan_action_normalized")
    assert normalized["requested_tool"] == "run_command"
    assert normalized["normalized_tool"] == "run_verification"
    assert normalized["check_id"] == "tests"
    assert not any(name == "action_blocked" for name, _ in events.events)


def test_execute_plan_exposes_only_current_step_side_effect_capability(tmp_path: Path) -> None:
    memory = inspected_memory(tmp_path)
    registry = create_default_registry(tmp_path)
    registry.plan_runtime.bind(memory)
    registry.plan_runtime.submit({"plan": plan_payload()})
    registry.plan_runtime.approve()
    agent = AgentController(
        model_client=FakeModelClient([]),
        tool_registry=registry,
    )

    schemas = agent._tool_schemas(WorkflowMode.EXECUTE)
    names = {schema["function"]["name"] for schema in schemas}

    assert "edit_file" in names
    assert "read_file" in names
    assert "update_plan" in names
    assert "run_command" not in names
    assert "run_verification" not in names
    assert "write_file" not in names


def test_stale_previous_step_tool_triggers_controller_scheduled_plan_check(
    tmp_path: Path,
) -> None:
    memory = inspected_memory(tmp_path)
    registry = create_default_registry(tmp_path)
    registry.plan_runtime.bind(memory)
    payload = plan_payload(operation="command")
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps[0]["target_files"] = []  # type: ignore[index]
    artifact = registry.plan_runtime.submit({"plan": payload})
    registry.plan_runtime.approve()
    registry.verification_runtime.bind(memory)
    registry.verification_runtime.register_plan(
        {
            "requirements": ["tests"],
            "checks": [
                {
                    "check_id": "tests",
                    "title": "Exact registered command",
                    "kind": "test",
                    "command": {"program": "true", "args": [], "cwd": "."},
                    "criteria": {"expected_exit_codes": [0]},
                    "required": True,
                    "provenance": "model",
                }
            ],
        }
    )
    stale_edit = ToolCall(
        id="stale-edit",
        name="edit_file",
        arguments={"path": "app.py", "snapshot_id": "old", "operations": []},
    )

    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def log(self, event_type: str, **data: object) -> None:
            self.events.append((event_type, data))

    events = RecordingSink()
    result = AgentController(
        model_client=FakeModelClient(
            [
                ModelResponse(tool_calls=[stale_edit]),
                ModelResponse(content="The approved plan and its checks are complete."),
            ]
        ),
        tool_registry=registry,
        event_logger=events,
    ).run("Continue the approved plan", memory=memory)

    assert result.status == "success"
    assert artifact.status == PlanStatus.COMPLETED
    assert memory.verification_results["tests"].status.value == "passed"
    blocked = [data for name, data in events.events if name == "action_blocked"]
    assert len(blocked) == 1
    assert blocked[0]["error_code"] == "plan_step_violation"
    scheduled = next(data for name, data in events.events if name == "plan_checks_scheduled")
    assert scheduled["step_id"] == "change"
    assert scheduled["check_ids"] == ["tests"]


def test_failed_controller_plan_check_blocks_same_revision_model_retry(
    tmp_path: Path,
) -> None:
    memory = inspected_memory(tmp_path)
    registry = create_default_registry(tmp_path)
    registry.plan_runtime.bind(memory)
    payload = plan_payload(operation="command")
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps[0]["target_files"] = []  # type: ignore[index]
    artifact = registry.plan_runtime.submit({"plan": payload})
    registry.plan_runtime.approve()
    registry.verification_runtime.bind(memory)
    registry.verification_runtime.register_plan(
        {
            "requirements": ["tests"],
            "checks": [
                {
                    "check_id": "tests",
                    "title": "Failing registered command",
                    "kind": "test",
                    "command": {"program": "false", "args": [], "cwd": "."},
                    "criteria": {"expected_exit_codes": [0]},
                    "required": True,
                    "provenance": "model",
                }
            ],
        }
    )
    stale_edit = ToolCall(
        id="stale-edit",
        name="edit_file",
        arguments={"path": "app.py", "snapshot_id": "old", "operations": []},
    )
    repeated = ToolCall(
        id="repeat-check",
        name="run_command",
        arguments={"command": "false", "cwd": ".", "timeout_seconds": 60},
    )
    repair_plan = ToolCall(
        id="repair-plan",
        name="update_plan",
        arguments={
            "reason": "The registered test failed and requires a source repair.",
            "plan": plan_payload(),
        },
    )

    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def log(self, event_type: str, **data: object) -> None:
            self.events.append((event_type, data))

    events = RecordingSink()
    result = AgentController(
        model_client=FakeModelClient(
                [
                    ModelResponse(tool_calls=[stale_edit]),
                    ModelResponse(tool_calls=[repeated]),
                    ModelResponse(tool_calls=[repair_plan]),
                ]
        ),
        tool_registry=registry,
        event_logger=events,
    ).run("Continue the approved plan", memory=memory)

    assert result.status == "plan_ready"
    assert artifact.current_step is not None
    assert artifact.current_step.status == PlanStepStatus.FAILED
    assert memory.plan_artifact is not None
    assert memory.plan_artifact.status == PlanStatus.READY
    blocked = [data for name, data in events.events if name == "action_blocked"]
    assert [item["error_code"] for item in blocked] == [
        "plan_step_violation",
        "verification_retry_without_change",
    ]
    assert sum(name == "verification_result" for name, _ in events.events) == 1
    assert any(name == "verification_repair_required" for name, _ in events.events)


def test_plan_update_preserves_completed_steps_and_clears_old_reflection(
    tmp_path: Path,
) -> None:
    memory = MemoryState(session_id="plan-update")
    memory.observation_events.append(observation("plan-update"))
    runtime = PlanRuntime(WorkspacePathPolicy(tmp_path))
    runtime.bind(memory)
    first = runtime.submit({"plan": plan_payload(operation="create", target="new.py")})
    runtime.approve()

    created_path = tmp_path / "new.py"
    created_path.write_text("value = 1\n", encoding="utf-8")
    created_snapshot = TextDocument.load(created_path).to_snapshot(relative_path="new.py")
    create_call = ToolCall(
        id="write-new",
        name="write_file",
        arguments={"path": "new.py", "content": "value = 1\n"},
    )
    runtime.start_action()
    runtime.observe_action(
        create_call,
        ToolResult(
            ok=True,
            output="created",
            metadata={
                "path": "new.py",
                "workspace_revision": 1,
                "snapshot_id": created_snapshot.snapshot_id,
            },
        ),
        evidence_ref="obs-write",
    )
    assert first.steps[0].status == PlanStepStatus.COMPLETED
    memory.workspace_revision = 1
    memory.current_snapshots["new.py"] = created_snapshot.snapshot_id

    revised = plan_payload(operation="create", target="new.py")
    steps = revised["steps"]
    assert isinstance(steps, list)
    steps.insert(
        1,
        {
            "step_id": "fix-selftest",
            "title": "Fix the selftest",
            "intent": "Replace the invalid selftest helper call",
            "evidence_refs": ["obs-read"],
            "operation": "edit",
            "target_files": ["new.py"],
            "verification_ids": ["tests"],
            "depends_on": ["change"],
            "risk": "low",
        },
    )
    steps[2]["depends_on"] = ["fix-selftest"]  # type: ignore[index]

    updated = runtime.submit(
        {"plan": revised},
        update_reason="The selftest references a missing helper",
    )

    assert updated.steps[0].status == PlanStepStatus.COMPLETED
    assert updated.steps[0].last_observation_ref == "obs-write"
    memory.reflection_required = True
    runtime.approve()
    assert updated.current_step is updated.steps[1]
    assert not memory.reflection_required


def test_plan_mode_hides_and_rejects_mutating_capabilities(tmp_path: Path) -> None:
    memory = MemoryState(session_id="plan-test")
    memory.last_agent_outcome = "success"
    memory.observation_events.append(observation())
    write = ToolCall(
        id="write-1",
        name="write_file",
        arguments={"path": "new.py", "content": "unsafe\n"},
    )
    submit = ToolCall(
        id="plan-1",
        name="submit_plan",
        arguments={"plan": plan_payload(operation="create", target="new.py")},
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[write]),
            ModelResponse(tool_calls=[submit]),
        ]
    )
    registry = create_default_registry(tmp_path)
    agent = AgentController(model_client=model, tool_registry=registry)

    result = agent.run("Plan a new file", memory=memory, workflow_mode=WorkflowMode.PLANNING)

    assert result.status == "plan_ready"
    assert not (tmp_path / "new.py").exists()
    visible_tools = {schema["function"]["name"] for schema in model.requests[0]["tools"]}
    assert visible_tools == PLANNING_TOOL_NAMES
    assert "write_file" not in visible_tools
    blocked = next(message for message in memory.messages if message.tool_call_id == "write-1")
    assert "plan_mode_violation" in (blocked.content or "")


def test_plan_mode_violation_is_typed() -> None:
    with pytest.raises(PlanModeViolation, match="run_command is unavailable"):
        PlanRuntime.ensure_tool_allowed("run_command")


def test_controller_reports_progress_checkpoint_and_remaining_plan_steps(
    tmp_path: Path,
) -> None:
    memory = inspected_memory(tmp_path)
    runtime = PlanRuntime(WorkspacePathPolicy(tmp_path))
    runtime.bind(memory)
    runtime.submit({"plan": plan_payload()})
    runtime.approve()
    model = FakeModelClient([ModelResponse(content="More plan work remains.")])
    registry = create_default_registry(tmp_path)

    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def log(self, event_type: str, **data: object) -> None:
            self.events.append((event_type, data))

    events = RecordingSink()
    agent = AgentController(
        model_client=model,
        tool_registry=registry,
        termination_policy=TerminationPolicy(TerminationConfig(max_steps=3)),
        event_logger=events,
    )

    agent.run("Execute the approved plan", memory=memory)

    session_start = next(data for name, data in events.events if name == "session_start")
    assert session_start["progress_checkpoint_window"] == 3
    assert session_start["remaining_plan_steps"] == 2
    assert session_start["next_step_checkpoint"] == 3


def test_adaptive_policy_only_preflights_clear_complexity_signals() -> None:
    policy = AutoPlanPolicy(AutoPlanMode.ADAPTIVE)

    assert policy.preflight_reason("Fix the typo in app.py") is None
    assert policy.preflight_reason("整体重构这个项目并迁移架构") is not None
    assert policy.preflight_reason("- inspect\n- design\n- implement\n- verify") is not None
    assert AutoPlanPolicy(AutoPlanMode.OFF).preflight_reason("整体重构项目") is None


def test_openai_plan_classifier_uses_isolated_workspace_metadata() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.request: dict[str, object] = {}

        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            self.request = kwargs
            message = SimpleNamespace(
                content=('{"decision":"plan","reason":"Greenfield architecture","confidence":0.94}')
            )
            usage = SimpleNamespace(prompt_tokens=180, completion_tokens=24)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    classifier = OpenAIPlanClassifier(
        ApiConfig(
            api_key=SecretStr("test-key"),
            base_url="https://example.com/v1",
            model="test-model",
            context_window_tokens=32_768,
        ),
        client=client,
    )
    workspace = WorkspacePlanningSummary(
        empty=True,
        sampled_files=0,
        sampled_directories=0,
        truncated=False,
        extensions=(),
        root_entries=(),
    )

    result = classifier.classify("写一个面向对象的 C++ 俄罗斯方块", workspace)

    assert result and result.decision == "plan"
    assert result.confidence == 0.94
    assert result.input_tokens == 180
    assert result.output_tokens == 24
    assert len(completions.request["messages"]) == 2  # type: ignore[arg-type]
    assert "tools" not in completions.request
    assert "max_tokens" not in completions.request
    assert completions.request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_workspace_planning_summary_is_bounded_and_ignores_dependencies(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "objects").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.cpp").write_text("ignored contents", encoding="utf-8")

    summary = summarize_workspace(tmp_path)

    assert not summary.empty
    assert summary.sampled_files == 1
    assert summary.sampled_directories == 1
    assert summary.extensions == (".cpp",)
    assert summary.root_entries == ("src",)


def test_adaptive_llm_classification_forces_plan_before_main_agent(tmp_path: Path) -> None:
    class FakeClassifier:
        def __init__(self) -> None:
            self.calls: list[tuple[str, WorkspacePlanningSummary]] = []

        def classify(
            self,
            task: str,
            workspace: WorkspacePlanningSummary,
        ) -> PlanClassification:
            self.calls.append((task, workspace))
            return PlanClassification(
                decision="plan",
                reason="Greenfield game with explicit object-oriented architecture",
                confidence=0.95,
            )

    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def log(self, event_type: str, **data: object) -> None:
            self.events.append((event_type, data))

    memory = inspected_memory(tmp_path)
    submit = ToolCall(
        id="submit",
        name="submit_plan",
        arguments={"plan": plan_payload()},
    )
    model = FakeModelClient([ModelResponse(tool_calls=[submit])])
    classifier = FakeClassifier()
    events = RecordingSink()
    agent = AgentController(
        model_client=model,
        tool_registry=create_default_registry(tmp_path),
        auto_plan_policy=AutoPlanPolicy(
            AutoPlanMode.ADAPTIVE,
            classifier_confidence_threshold=0.70,
        ),
        plan_classifier=classifier,
        event_logger=events,
    )

    result = agent.run("写一个cpp的俄罗斯方块，必须完全面向对象", memory=memory)

    assert result.status == "plan_ready"
    assert len(classifier.calls) == 1
    assert {schema["function"]["name"] for schema in model.requests[0]["tools"]} == (
        PLANNING_TOOL_NAMES
    )
    reviewed = next(data for name, data in events.events if name == "auto_plan_reviewed")
    assert reviewed["decision"] == "plan"
    assert reviewed["applied"] is True
    started = next(data for name, data in events.events if name == "auto_plan_started")
    assert started["source"] == "llm_classifier"


def test_adaptive_classifier_failure_preserves_model_requested_fallback(tmp_path: Path) -> None:
    class FailingClassifier:
        def classify(self, task: str, workspace: WorkspacePlanningSummary) -> None:
            del task, workspace
            raise TimeoutError("classifier timed out")

    request = ToolCall(
        id="request-plan",
        name="request_plan",
        arguments={
            "reason": "The implementation boundary is uncertain",
            "expected_scope": "Inspect and prepare a plan",
        },
    )
    submit = ToolCall(
        id="submit",
        name="submit_plan",
        arguments={"plan": plan_payload()},
    )
    model = FakeModelClient(
        [ModelResponse(tool_calls=[request]), ModelResponse(tool_calls=[submit])]
    )
    events: list[tuple[str, dict[str, object]]] = []

    class RecordingSink:
        def log(self, event_type: str, **data: object) -> None:
            events.append((event_type, data))

    agent = AgentController(
        model_client=model,
        tool_registry=create_default_registry(tmp_path),
        auto_plan_policy=AutoPlanPolicy(AutoPlanMode.ADAPTIVE),
        plan_classifier=FailingClassifier(),
        event_logger=RecordingSink(),
    )

    result = agent.run("Determine the implementation scope", memory=inspected_memory(tmp_path))

    assert result.status == "plan_ready"
    failed = next(data for name, data in events if name == "auto_plan_review_failed")
    assert failed["reason"] == "TimeoutError"
    first_tools = {schema["function"]["name"] for schema in model.requests[0]["tools"]}
    assert "request_plan" in first_tools


def test_always_auto_plan_enters_read_only_workflow_before_first_model_call(
    tmp_path: Path,
) -> None:
    memory = inspected_memory(tmp_path)
    submit = ToolCall(
        id="submit",
        name="submit_plan",
        arguments={"plan": plan_payload()},
    )
    model = FakeModelClient([ModelResponse(tool_calls=[submit])])
    registry = create_default_registry(tmp_path)
    agent = AgentController(
        model_client=model,
        tool_registry=registry,
        auto_plan_policy=AutoPlanPolicy(AutoPlanMode.ALWAYS),
    )

    result = agent.run("Fix app.py", memory=memory)

    assert result.status == "plan_ready"
    visible_tools = {schema["function"]["name"] for schema in model.requests[0]["tools"]}
    assert visible_tools == PLANNING_TOOL_NAMES
    assert "write_file" not in visible_tools


def test_model_can_request_adaptive_plan_before_any_action(tmp_path: Path) -> None:
    memory = inspected_memory(tmp_path)
    request = ToolCall(
        id="request-plan",
        name="request_plan",
        arguments={
            "reason": "The change spans uncertain boundaries",
            "expected_scope": "Inspect dependencies and prepare bounded file steps",
        },
    )
    coissued_write = ToolCall(
        id="unsafe-write",
        name="write_file",
        arguments={"path": "should-not-exist.py", "content": "unsafe = True\n"},
    )
    submit = ToolCall(
        id="submit",
        name="submit_plan",
        arguments={"plan": plan_payload()},
    )
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[request, coissued_write]),
            ModelResponse(tool_calls=[submit]),
        ]
    )
    registry = create_default_registry(tmp_path)
    agent = AgentController(
        model_client=model,
        tool_registry=registry,
        auto_plan_policy=AutoPlanPolicy(AutoPlanMode.ADAPTIVE),
    )

    result = agent.run("Decide the safe implementation scope", memory=memory)

    assert result.status == "plan_ready"
    first_tools = {schema["function"]["name"] for schema in model.requests[0]["tools"]}
    second_tools = {schema["function"]["name"] for schema in model.requests[1]["tools"]}
    assert "request_plan" in first_tools
    assert "submit_plan" not in first_tools
    assert second_tools == PLANNING_TOOL_NAMES
    assert not (tmp_path / "should-not-exist.py").exists()
    request_result = next(item for item in memory.messages if item.tool_call_id == "request-plan")
    assert "read-only Plan Mode" in (request_result.content or "")
    blocked_write = next(item for item in memory.messages if item.tool_call_id == "unsafe-write")
    assert "were not executed" in (blocked_write.content or "")


def test_auto_plan_off_hides_model_transition_tool(tmp_path: Path) -> None:
    model = FakeModelClient([ModelResponse(content="No change is required.")])
    registry = create_default_registry(tmp_path)
    agent = AgentController(
        model_client=model,
        tool_registry=registry,
        auto_plan_policy=AutoPlanPolicy(AutoPlanMode.OFF),
    )

    result = agent.run("Inspect the current state", memory=MemoryState(session_id="off"))

    assert result.status == "success"
    visible_tools = {schema["function"]["name"] for schema in model.requests[0]["tools"]}
    assert "request_plan" not in visible_tools
