from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from repo_rivet.agent.controller import AgentController
from repo_rivet.cli import cli
from repo_rivet.editing.document import TextDocument
from repo_rivet.llm.base import ModelResponse
from repo_rivet.memory.models import MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.planning.models import PlanStatus, WorkflowMode
from repo_rivet.planning.runtime import PlanRuntime
from repo_rivet.reasoning.models import ObservationEvent
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.skills.errors import SkillStaleError, SkillValidationError
from repo_rivet.skills.models import SkillActivation
from repo_rivet.skills.registry import SkillRegistry
from repo_rivet.skills.requirements import SkillRequirementEvaluator
from repo_rivet.skills.runtime import SkillRuntime
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.tools.registry import create_default_registry
from repo_rivet.verification.models import (
    CommandSpec,
    SuccessCriteria,
    VerificationCheck,
    VerificationKind,
    VerificationPlan,
    VerificationResult,
    VerificationStatus,
)
from tests.fakes import FakeModelClient, FakeToolRegistry


def write_skill(
    root: Path,
    skill_id: str = "sample-skill",
    *,
    body: str = "# Procedure\n\nInspect the target.",
    extra: str = "",
    requested_tools: str = "  - read_file",
    before_finish: str = "  before_finish: []",
    automatic: bool = False,
) -> Path:
    path = root / skill_id / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""---
schema_version: 1
id: {skill_id}
name: Sample Skill
version: 1.0.0
summary: A small test skill.
activation:
  explicit: {str(not automatic).lower()}
  automatic: {str(automatic).lower()}
compatible_modes: [plan, execute]
requested_tools:
{requested_tools}
requirements:
  before_edit: []
{before_finish}
limits:
  max_prompt_tokens: 1000
  max_active_support_skills: 0
{extra}---

{body}
""",
        encoding="utf-8",
    )
    return path


def registry_for(tmp_path: Path) -> SkillRegistry:
    return SkillRegistry(
        system_root=tmp_path / "system",
        global_root=tmp_path / "global",
    )


def behavior_plan(call_id: str, *, kind: str = "behavior") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="register_verification",
        arguments={
            "requirements": ["selftest"],
            "checks": [
                {
                    "check_id": "selftest",
                    "title": "Run deterministic self-test",
                    "kind": kind,
                    "command": {"program": "pytest", "args": ["--version"]},
                    "criteria": {
                        "expected_exit_codes": [0],
                        "stdout_contains": ["ok"],
                    },
                    "required": True,
                    "provenance": "model",
                }
            ],
        },
    )


def passed_check(check_id: str = "selftest", revision: int = 0) -> ToolResult:
    now = datetime.now(UTC).isoformat()
    return ToolResult(
        ok=True,
        output="ok",
        metadata={
            "exit_code": 0,
            "verification_result": {
                "check_id": check_id,
                "status": "passed",
                "workspace_revision": revision,
                "exit_code": 0,
                "reasons": ["all registered success criteria passed"],
                "started_at": now,
                "finished_at": now,
            },
        },
    )


def test_registry_indexes_metadata_and_loads_body_only_when_selected(tmp_path: Path) -> None:
    path = write_skill(tmp_path / "system")
    registry = registry_for(tmp_path)

    metadata = registry.discover()[0]
    assert metadata.manifest.id == "sample-skill"
    assert not hasattr(metadata, "body")

    original = registry.load("sample-skill")
    path.write_text(path.read_text(encoding="utf-8").replace("Inspect", "Review"), encoding="utf-8")
    changed = registry.load("sample-skill")
    assert "Review the target" in changed.body
    assert changed.content_hash != original.content_hash


def test_global_skill_cannot_shadow_a_system_skill(tmp_path: Path) -> None:
    write_skill(tmp_path / "system", "reserved-skill", body="# System\n\nSYSTEM")
    write_skill(tmp_path / "global", "reserved-skill", body="# Global\n\nGLOBAL")
    registry = registry_for(tmp_path)

    assert registry.metadata("reserved-skill").source.value == "system"
    assert registry.global_skills() == ()
    assert any(key == "global:reserved-skill" for key, _error in registry.discovery_errors())


def test_runtime_eagerly_loads_all_system_skills_without_narrowing_tools(
    tmp_path: Path,
) -> None:
    write_skill(tmp_path / "system", "system-one", body="# One\n\nSYSTEM_ONE", automatic=True)
    write_skill(tmp_path / "system", "system-two", body="# Two\n\nSYSTEM_TWO", automatic=True)
    runtime = SkillRuntime(
        registry_for(tmp_path),
        known_tools={"read_file", "edit_file"},
    )

    assert [bundle.manifest.id for bundle in runtime.system] == ["system-one", "system-two"]
    assert runtime.active is None
    assert runtime.allowed_tool_names({"read_file", "edit_file"}) == {
        "read_file",
        "edit_file",
    }

    model = FakeModelClient([ModelResponse(content="Done.")])
    controller = AgentController(
        model_client=model,
        tool_registry=create_default_registry(tmp_path),
        skill_runtime=runtime,
    )
    memory = MemoryState(session_id="sys")
    result = controller.run("Inspect only what is relevant", memory=memory)

    assert result.status == "success"
    request = str(model.requests[0]["messages"])
    assert "SYSTEM_ONE" in request
    assert "SYSTEM_TWO" in request
    assert "requirements_enforced" in request
    assert [pin.id for pin in memory.system_skills] == [
        "system-one",
        "system-two",
    ]


def test_legacy_skill_sources_migrate_to_system_and_global_layers() -> None:
    base_pin = {
        "id": "legacy-skill",
        "version": "1.0.0",
        "content_hash": "abc",
        "activation": "explicit",
    }

    former_system = MemoryState.model_validate(
        {"session_id": "old-system", "active_skill": {**base_pin, "source": "builtin"}}
    )
    former_global = MemoryState.model_validate(
        {"session_id": "old-global", "active_skill": {**base_pin, "source": "user"}}
    )

    assert former_system.active_skill is None
    assert former_global.active_skill is not None
    assert former_global.active_skill.source.value == "global"


def test_cli_lists_and_updates_selected_session_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPORIVET_HOME", str(tmp_path / "home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    console_buffer = StringIO()
    console = Console(file=console_buffer, force_terminal=False, color_system=None)

    assert cli(["skill", "list"], console=console) == 0
    assert "RepoRivet Skills" in console_buffer.getvalue()
    assert cli(["skill", "show", "repository-onboarding"], console=console) == 0
    assert "ID: repository-onboarding" in console_buffer.getvalue()
    drafts = tmp_path / "drafts"
    assert (
        cli(
            [
                "skill",
                "init",
                "global-review",
                "--output",
                str(drafts),
                "--tool",
                "read_file",
            ],
            console=console,
        )
        == 0
    )
    assert cli(["skill", "install", str(drafts / "global-review")], console=console) == 0
    assert cli(["session", "new", "--workspace", str(workspace)], console=console) == 0
    assert (
        cli(
            [
                "skill",
                "use",
                "repository-onboarding",
                "--workspace",
                str(workspace),
            ],
            console=console,
        )
        == 2
    )
    assert "already loaded" in console_buffer.getvalue()
    assert (
        cli(
            [
                "skill",
                "use",
                "global-review",
                "--workspace",
                str(workspace),
            ],
            console=console,
        )
        == 0
    )
    loaded = MemoryStore(next((tmp_path / "home" / "sessions").iterdir())).load_state()
    assert loaded.active_skill is not None
    assert loaded.active_skill.id == "global-review"

    assert cli(["skill", "clear", "--workspace", str(workspace)], console=console) == 0
    cleared = MemoryStore(next((tmp_path / "home" / "sessions").iterdir())).load_state()
    assert cleared.active_skill is None
    assert cli(["skill", "uninstall", "global-review"], console=console) == 0
    assert not (tmp_path / "home" / "skills" / "global-review").exists()


def test_loader_rejects_executable_hooks_and_unknown_requirements(tmp_path: Path) -> None:
    write_skill(tmp_path / "system", extra="hooks: [run.py]\n")
    with pytest.raises(SkillValidationError, match="hooks"):
        registry_for(tmp_path).load("sample-skill")

    path = tmp_path / "system" / "sample-skill" / "SKILL.md"
    path.unlink()
    path.parent.rmdir()
    write_skill(
        tmp_path / "system",
        before_finish="  before_finish: [model_invented_check]",
    )
    with pytest.raises(SkillValidationError, match="unknown requirements"):
        registry_for(tmp_path).load("sample-skill")


def test_activation_pins_content_and_rejects_changed_session_skill(tmp_path: Path) -> None:
    path = write_skill(tmp_path / "global")
    registry = registry_for(tmp_path)
    memory = MemoryState(session_id="skill-session")
    runtime = SkillRuntime(registry, known_tools={"read_file", "edit_file"})

    bundle = runtime.activate(memory, "sample-skill", activation=SkillActivation.EXPLICIT)
    assert memory.active_skill is not None
    assert memory.active_skill.content_hash == bundle.content_hash

    store = MemoryStore(tmp_path / "session")
    store.save_state(memory, status="paused")
    restored = store.load_state()
    assert restored.active_skill == memory.active_skill

    path.write_text(path.read_text(encoding="utf-8").replace("Inspect", "Review"), encoding="utf-8")
    with pytest.raises(SkillStaleError, match="changed after activation"):
        SkillRuntime(registry, known_tools={"read_file"}).restore(restored)


def test_skill_tools_are_intersected_with_execute_and_plan_modes(tmp_path: Path) -> None:
    write_skill(
        tmp_path / "global",
        requested_tools="  - read_file\n  - edit_file\n  - run_verification",
    )
    registry = create_default_registry(tmp_path)
    skill_runtime = SkillRuntime(registry_for(tmp_path), known_tools=set(registry.names))
    memory = MemoryState(session_id="intersection")
    skill_runtime.activate(memory, "sample-skill")
    controller = AgentController(
        model_client=object(),  # type: ignore[arg-type]
        tool_registry=registry,
        skill_runtime=skill_runtime,
    )

    execute_names = {
        schema["function"]["name"] for schema in controller._tool_schemas(WorkflowMode.EXECUTE)
    }
    plan_names = {
        schema["function"]["name"] for schema in controller._tool_schemas(WorkflowMode.PLANNING)
    }
    assert "edit_file" in execute_names
    assert "list_files" not in execute_names
    assert "read_file" in plan_names
    assert "edit_file" not in plan_names
    assert "submit_plan" in plan_names
    assert "record_decision" in execute_names


def test_controller_injects_only_active_body_and_rejects_unrequested_tool(tmp_path: Path) -> None:
    write_skill(tmp_path / "global", body="# Unique\n\nACTIVE_SKILL_BODY")
    registry = create_default_registry(tmp_path)
    skill_runtime = SkillRuntime(registry_for(tmp_path), known_tools=set(registry.names))
    memory = MemoryState(session_id="controller-skill")
    skill_runtime.activate(memory, "sample-skill")
    model = FakeModelClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="forbidden",
                        name="write_file",
                        arguments={"path": "created.txt", "content": "unsafe"},
                    )
                ]
            ),
            ModelResponse(content="Inspection complete."),
        ]
    )
    controller = AgentController(
        model_client=model,
        tool_registry=registry,
        skill_runtime=skill_runtime,
    )

    result = controller.run("Inspect", memory=memory)

    assert result.status == "success"
    assert not (tmp_path / "created.txt").exists()
    assert "ACTIVE_SKILL_BODY" in str(model.requests[0]["messages"])
    for request in model.requests:
        skill_messages = [
            message
            for message in request["messages"]
            if "ACTIVE_SKILL_BODY" in str(message.get("content"))
        ]
        assert len(skill_messages) == 1
        assert request["messages"][1] == skill_messages[0]
    assert "ACTIVE_SKILL_BODY" not in str([message.content for message in memory.messages])
    tool_result = next(
        message
        for message in model.requests[1]["messages"]
        if message.get("role") == "tool" and message.get("tool_call_id") == "forbidden"
    )
    assert "skill_tool_violation" in str(tool_result["content"])


def test_global_skill_rejects_plan_without_required_verification_kind(tmp_path: Path) -> None:
    write_skill(
        tmp_path / "global",
        "behavior-required",
        requested_tools="  - run_verification",
        before_finish="  before_finish: [required_behavior_checks_passed]",
    )
    runtime = SkillRuntime(
        registry_for(tmp_path),
        known_tools={"run_verification"},
    )
    memory = MemoryState(session_id="skill-plan-coverage")
    runtime.activate(memory, "behavior-required")
    model = FakeModelClient(
        [
            ModelResponse(tool_calls=[behavior_plan("wrong-plan", kind="test")]),
            ModelResponse(tool_calls=[behavior_plan("correct-plan")]),
            ModelResponse(content="Behavior is verified."),
        ]
    )
    tools = FakeToolRegistry([passed_check()])
    controller = AgentController(
        model_client=model,
        tool_registry=tools,  # type: ignore[arg-type]
        skill_runtime=runtime,
    )

    result = controller.run("verify behavior", memory=memory)

    assert result.status == "success"
    rejected = next(
        message
        for message in memory.messages
        if message.role == "tool" and message.tool_call_id == "wrong-plan"
    )
    assert "missing required check kinds: behavior" in (rejected.content or "")
    assert [call.name for call in tools.calls] == ["run_verification"]


def test_existing_plan_missing_skill_kind_enters_bounded_plan_revision(tmp_path: Path) -> None:
    write_skill(
        tmp_path / "global",
        "behavior-required",
        requested_tools="  - run_verification",
        before_finish="  before_finish: [required_behavior_checks_passed]",
    )
    runtime = SkillRuntime(
        registry_for(tmp_path),
        known_tools={"run_verification"},
    )
    memory = MemoryState(session_id="legacy-skill-plan-gap")
    runtime.activate(memory, "behavior-required")
    memory.verification_plan = VerificationPlan(
        plan_id="old-plan",
        requirements=["selftest"],
        checks=[
            VerificationCheck(
                check_id="selftest",
                title="Run self-test",
                kind=VerificationKind.TEST,
                command=CommandSpec(program="pytest", args=["--version"]),
                criteria=SuccessCriteria(stdout_contains=["ok"]),
                provenance="model",
            )
        ],
    )
    now = datetime.now(UTC)
    memory.verification_results["selftest"] = VerificationResult(
        check_id="selftest",
        status=VerificationStatus.PASSED,
        workspace_revision=0,
        exit_code=0,
        started_at=now,
        finished_at=now,
    )
    model = FakeModelClient(
        [
            ModelResponse(content="Task complete."),
            ModelResponse(tool_calls=[behavior_plan("replacement-plan")]),
            ModelResponse(content="Task complete with behavior evidence."),
        ]
    )
    tools = FakeToolRegistry([passed_check()])
    controller = AgentController(
        model_client=model,
        tool_registry=tools,  # type: ignore[arg-type]
        skill_runtime=runtime,
    )

    result = controller.run("finish the existing task", memory=memory)

    assert result.status == "success"
    assert len(model.requests) == 3
    assert any(
        "missing verification kinds: behavior" in str(message.get("content"))
        for message in model.requests[1]["messages"]
    )
    assert memory.verification_plan is not None
    assert memory.verification_plan.checks[0].kind == VerificationKind.BEHAVIOR
    assert memory.verification_plan_revision_required is False


def test_unsatisfied_non_verification_skill_requirement_stops_after_bounded_recovery(
    tmp_path: Path,
) -> None:
    write_skill(
        tmp_path / "global",
        "diff-required",
        requested_tools="  - git_diff",
        before_finish="  before_finish: [git_diff_reviewed]",
    )
    runtime = SkillRuntime(registry_for(tmp_path), known_tools={"git_diff"})
    memory = MemoryState(session_id="bounded-skill-completion")
    runtime.activate(memory, "diff-required")
    memory.modified_files.add("app.py")
    memory.observation_events.append(
        ObservationEvent(
            event_id="obs-edit",
            session_id=memory.session_id,
            step=1,
            tool_call_id="edit",
            tool_name="edit_file",
            ok=True,
            result_summary="Edited app.py.",
            affected_paths=["app.py"],
        )
    )
    memory.verification_plan = VerificationPlan(
        plan_id="verified",
        checks=[
            VerificationCheck(
                check_id="tests",
                title="Tests",
                kind=VerificationKind.TEST,
                command=CommandSpec(program="pytest"),
                provenance="model",
            )
        ],
    )
    now = datetime.now(UTC)
    memory.verification_results["tests"] = VerificationResult(
        check_id="tests",
        status=VerificationStatus.PASSED,
        workspace_revision=0,
        exit_code=0,
        started_at=now,
        finished_at=now,
    )
    model = FakeModelClient([ModelResponse(content="Done.") for _item in range(3)])
    controller = AgentController(
        model_client=model,
        tool_registry=FakeToolRegistry([]),  # type: ignore[arg-type]
        skill_runtime=runtime,
    )

    result = controller.run("finish without reviewing the diff", memory=memory)

    assert result.status == "incomplete"
    assert result.reason is not None
    assert "git_diff_reviewed" in result.reason
    assert len(model.requests) == 3


def test_plan_artifact_pins_skill_and_becomes_stale_after_skill_change(tmp_path: Path) -> None:
    write_skill(tmp_path / "system", "system-guidance", automatic=True)
    write_skill(tmp_path / "global", "first-skill")
    write_skill(tmp_path / "global", "second-skill")
    skill_runtime = SkillRuntime(
        registry_for(tmp_path),
        known_tools={"read_file", "edit_file"},
    )
    memory = MemoryState(session_id="plan-skill")
    skill_runtime.sync_system(memory)
    skill_runtime.activate(memory, "first-skill")
    source = tmp_path / "app.py"
    source.write_text("value = 1\n", encoding="utf-8")
    snapshot = TextDocument.load(source).to_snapshot(relative_path="app.py")
    memory.current_snapshots["app.py"] = snapshot.snapshot_id
    memory.observation_events.append(
        ObservationEvent(
            event_id="obs-read",
            session_id=memory.session_id,
            step=1,
            tool_call_id="read",
            tool_name="read_file",
            ok=True,
            result_summary="Read app.py:1.",
            affected_paths=["app.py"],
        )
    )
    plan = PlanRuntime(WorkspacePathPolicy(tmp_path))
    plan.bind(memory)
    artifact = plan.submit(
        {
            "plan": {
                "goal": "Change the value",
                "evidence_refs": ["obs-read"],
                "steps": [
                    {
                        "step_id": "edit",
                        "title": "Edit",
                        "intent": "Change inspected value",
                        "evidence_refs": ["obs-read"],
                        "operation": "edit",
                        "target_files": ["app.py"],
                        "verification_ids": ["tests"],
                        "risk": "low",
                    },
                    {
                        "step_id": "verify",
                        "title": "Verify",
                        "intent": "Run tests",
                        "evidence_refs": ["obs-read"],
                        "operation": "verify",
                        "verification_ids": ["tests"],
                        "depends_on": ["edit"],
                        "risk": "low",
                    },
                ],
                "verification": [
                    {
                        "check_id": "tests",
                        "title": "Tests",
                        "success_criteria": "Tests pass",
                    }
                ],
            }
        }
    )
    assert artifact.system_skills == memory.system_skills
    assert artifact.skill == memory.active_skill

    system_path = tmp_path / "system" / "system-guidance" / "SKILL.md"
    system_path.write_text(
        system_path.read_text(encoding="utf-8").replace("Inspect the target", "Review the target"),
        encoding="utf-8",
    )
    changed_runtime = SkillRuntime(
        registry_for(tmp_path),
        known_tools={"read_file", "edit_file"},
    )
    changed_runtime.sync_system(memory)
    assert any("system Skills changed" in reason for reason in plan.stale_reasons(artifact))

    skill_runtime.activate(memory, "second-skill")
    assert artifact.status == PlanStatus.STALE
    assert any("global Skill changed" in reason for reason in plan.stale_reasons(artifact))


def test_fixed_completion_requirement_uses_current_typed_test_result() -> None:
    memory = MemoryState(session_id="requirements", workspace_revision=2)
    memory.verification_plan = VerificationPlan(
        plan_id="verify",
        checks=[
            VerificationCheck(
                check_id="tests",
                title="Tests",
                kind=VerificationKind.TEST,
                command=CommandSpec(program="pytest"),
                criteria=SuccessCriteria(),
                provenance="user",
            )
        ],
    )
    evaluator = SkillRequirementEvaluator()
    failed = evaluator.before_finish(memory, ["required_tests_passed"])
    assert not failed.complete

    now = datetime.now(UTC)
    memory.verification_results["tests"] = VerificationResult(
        check_id="tests",
        status=VerificationStatus.PASSED,
        workspace_revision=2,
        exit_code=0,
        started_at=now,
        finished_at=now,
    )
    passed = evaluator.before_finish(memory, ["required_tests_passed"])
    assert passed.complete
