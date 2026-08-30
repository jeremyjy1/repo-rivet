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
from repo_rivet.skills.runtime import SkillRuntime
from repo_rivet.tools.registry import create_default_registry
from tests.fakes import FakeModelClient


def write_skill(
    root: Path,
    skill_id: str = "sample-skill",
    *,
    body: str = "# Procedure\n\nInspect the target.",
    extra: str = "",
    description: str = "Inspect a target. Use for inspect target tasks.",
) -> Path:
    path = root / skill_id / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""---
name: {skill_id}
description: {description}
metadata:
  version: "1.0.0"
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


def test_registry_indexes_metadata_and_loads_body_only_when_selected(tmp_path: Path) -> None:
    path = write_skill(tmp_path / "system")
    registry = registry_for(tmp_path)

    metadata = registry.discover()[0]
    assert metadata.manifest.name == "sample-skill"
    assert metadata.qualified_id == "system:sample-skill"
    assert not hasattr(metadata, "body")

    original = registry.load("sample-skill")
    path.write_text(path.read_text(encoding="utf-8").replace("Inspect", "Review"), encoding="utf-8")
    changed = registry.load("sample-skill")
    assert "Review the target" in changed.body
    assert changed.content_hash != original.content_hash


def test_same_name_skills_are_qualified_instead_of_shadowed(tmp_path: Path) -> None:
    write_skill(tmp_path / "system", "reserved-skill", body="# System\n\nSYSTEM")
    write_skill(tmp_path / "global", "reserved-skill", body="# Global\n\nGLOBAL")
    registry = registry_for(tmp_path)

    assert registry.metadata("system:reserved-skill").source.value == "system"
    assert registry.metadata("global:reserved-skill").source.value == "global"
    with pytest.raises(SkillValidationError, match="ambiguous"):
        registry.metadata("reserved-skill")


def test_runtime_indexes_system_skills_without_loading_unmatched_bodies(
    tmp_path: Path,
) -> None:
    write_skill(
        tmp_path / "system",
        "system-one",
        body="# One\n\nSYSTEM_ONE",
        description="Inspect repositories. Use for special inspection tasks.",
    )
    write_skill(
        tmp_path / "system",
        "system-two",
        body="# Two\n\nSYSTEM_TWO",
        description="Publish releases. Use for release publishing tasks.",
    )
    runtime = SkillRuntime(registry_for(tmp_path))

    assert [item.manifest.name for item in runtime.system] == ["system-one", "system-two"]
    assert all(not hasattr(item, "body") for item in runtime.system)
    assert runtime.active is None

    model = FakeModelClient([ModelResponse(content="Done.")])
    controller = AgentController(
        model_client=model,
        tool_registry=create_default_registry(tmp_path),
        skill_runtime=runtime,
    )
    memory = MemoryState(session_id="sys")
    result = controller.run("Perform a special inspection", memory=memory)

    assert result.status == "success"
    request = str(model.requests[0]["messages"])
    assert "SYSTEM_ONE" in request
    assert "SYSTEM_TWO" not in request
    assert [pin.id for pin in memory.system_skills] == [
        "system:system-one",
        "system:system-two",
    ]


def test_runtime_injects_only_system_skills_matching_the_task(tmp_path: Path) -> None:
    write_skill(
        tmp_path / "system",
        "matching-skill",
        body="# Matching\n\nMATCHING_BODY",
        description="Use a special workflow. Use for special workflow requests.",
    )
    write_skill(
        tmp_path / "system",
        "unrelated-skill",
        body="# Unrelated\n\nUNRELATED_BODY",
        description="Use a different process. Use for unrelated release requests.",
    )
    runtime = SkillRuntime(registry_for(tmp_path))

    assert [item.manifest.name for item in runtime.system] == [
        "matching-skill",
        "unrelated-skill",
    ]
    assert [bundle.manifest.name for bundle in runtime.system_for_task("Use SPECIAL workflow")] == [
        "matching-skill"
    ]

    model = FakeModelClient([ModelResponse(content="Done.")])
    controller = AgentController(
        model_client=model,
        tool_registry=create_default_registry(tmp_path),
        skill_runtime=runtime,
    )
    result = controller.run("Use special workflow", memory=MemoryState(session_id="matched"))

    assert result.status == "success"
    request = str(model.requests[0]["messages"])
    assert "MATCHING_BODY" in request
    assert "UNRELATED_BODY" not in request


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
    assert former_global.active_skill.id == "global:legacy-skill"


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
    assert "ID: system:repository-onboarding" in console_buffer.getvalue()
    drafts = tmp_path / "drafts"
    assert (
        cli(
            [
                "skill",
                "init",
                "global-review",
                "--output",
                str(drafts),
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
    assert "automatically routed" in console_buffer.getvalue()
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
    assert loaded.active_skill.id == "global:global-review"

    assert cli(["skill", "clear", "--workspace", str(workspace)], console=console) == 0
    cleared = MemoryStore(next((tmp_path / "home" / "sessions").iterdir())).load_state()
    assert cleared.active_skill is None
    assert cli(["skill", "uninstall", "global-review"], console=console) == 0
    assert not (tmp_path / "home" / "skills" / "global-review").exists()


def test_loader_retains_unknown_metadata_without_treating_it_as_behavior(tmp_path: Path) -> None:
    write_skill(
        tmp_path / "system",
        extra="hooks: [run.py]\nrequirements:\n  before_finish: [invented_check]\n",
    )
    bundle = registry_for(tmp_path).load("sample-skill")

    assert bundle.manifest.model_extra == {
        "hooks": ["run.py"],
        "requirements": {"before_finish": ["invented_check"]},
    }
    assert bundle.script_files == ()


def test_activation_pins_content_and_rejects_changed_session_skill(tmp_path: Path) -> None:
    path = write_skill(tmp_path / "global")
    registry = registry_for(tmp_path)
    memory = MemoryState(session_id="skill-session")
    runtime = SkillRuntime(registry)

    bundle = runtime.activate(memory, "sample-skill", activation=SkillActivation.EXPLICIT)
    assert memory.active_skill is not None
    assert memory.active_skill.content_hash == bundle.content_hash

    store = MemoryStore(tmp_path / "session")
    store.save_state(memory, status="paused")
    restored = store.load_state()
    assert restored.active_skill == memory.active_skill

    path.write_text(path.read_text(encoding="utf-8").replace("Inspect", "Review"), encoding="utf-8")
    with pytest.raises(SkillStaleError, match="changed after activation"):
        SkillRuntime(registry).restore(restored)


def test_skill_metadata_cannot_change_execute_or_plan_capabilities(tmp_path: Path) -> None:
    write_skill(
        tmp_path / "global",
        extra="allowed-tools: [read_file]\n",
    )
    registry = create_default_registry(tmp_path)
    skill_runtime = SkillRuntime(registry_for(tmp_path))
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
    assert "list_files" in execute_names
    assert "read_file" in plan_names
    assert "edit_file" not in plan_names
    assert "submit_plan" in plan_names
    assert "record_decision" in execute_names


def test_controller_injects_active_body_without_granting_metadata_authority(tmp_path: Path) -> None:
    write_skill(
        tmp_path / "global",
        body="# Unique\n\nACTIVE_SKILL_BODY",
        extra="allowed-tools: [read_file]\n",
    )
    registry = create_default_registry(tmp_path)
    skill_runtime = SkillRuntime(registry_for(tmp_path))
    memory = MemoryState(session_id="controller-skill")
    skill_runtime.activate(memory, "sample-skill")
    model = FakeModelClient([ModelResponse(content="Inspection complete.")])
    controller = AgentController(
        model_client=model,
        tool_registry=registry,
        skill_runtime=skill_runtime,
    )

    result = controller.run("Inspect", memory=memory)

    assert result.status == "success"
    assert "ACTIVE_SKILL_BODY" in str(model.requests[0]["messages"])
    assert "write_file" in {schema["function"]["name"] for schema in model.requests[0]["tools"]}
    for request in model.requests:
        skill_messages = [
            message
            for message in request["messages"]
            if "ACTIVE_SKILL_BODY" in str(message.get("content"))
        ]
        assert len(skill_messages) == 1
        assert request["messages"][1] == skill_messages[0]
    assert "ACTIVE_SKILL_BODY" not in str([message.content for message in memory.messages])


def test_plan_artifact_pins_skill_and_becomes_stale_after_skill_change(tmp_path: Path) -> None:
    write_skill(tmp_path / "system", "system-guidance")
    write_skill(tmp_path / "global", "first-skill")
    write_skill(tmp_path / "global", "second-skill")
    skill_runtime = SkillRuntime(registry_for(tmp_path))
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
    changed_runtime = SkillRuntime(registry_for(tmp_path))
    changed_runtime.sync_system(memory)
    assert any("system Skills changed" in reason for reason in plan.stale_reasons(artifact))

    skill_runtime.activate(memory, "second-skill")
    assert artifact.status == PlanStatus.STALE
    assert any("global Skill changed" in reason for reason in plan.stale_reasons(artifact))
