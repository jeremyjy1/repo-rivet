from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from repo_rivet.cli import cli
from repo_rivet.skills.authoring import (
    convert_skill,
    create_skill,
    install_skill,
    uninstall_skill,
    validate_skill,
)
from repo_rivet.skills.errors import SkillValidationError
from repo_rivet.skills.loader import load_bundle, load_metadata
from repo_rivet.skills.models import SkillSource
from repo_rivet.skills.registry import SkillRegistry
from repo_rivet.tools.registry import create_default_registry


def known_tools(tmp_path: Path) -> set[str]:
    return set(create_default_registry(tmp_path).names)


def test_packaged_system_skills_are_valid_guidance_without_unenforced_contracts(
    tmp_path: Path,
) -> None:
    system_root = Path(__file__).parents[1] / "src" / "repo_rivet" / "builtin_skills"
    registry = SkillRegistry(
        system_root=system_root,
        global_root=tmp_path / "global-skills",
    )
    tools = known_tools(tmp_path)

    skills = {item.manifest.id: load_bundle(item) for item in registry.system_skills()}

    assert set(skills) == {
        "repository-onboarding",
        "skill-authoring",
        "test-failure-fix",
    }
    for bundle in skills.values():
        validate_skill(bundle.path, known_tools=tools)
        assert bundle.manifest.activation.automatic is True
        assert bundle.manifest.activation.explicit is False
        assert bundle.manifest.requirements.before_edit == []
        assert bundle.manifest.requirements.before_finish == []
        assert bundle.manifest.verification_profiles == []

    onboarding = skills["repository-onboarding"].manifest
    assert onboarding.triggers.project_markers == []
    assert "理解项目" in onboarding.triggers.keywords
    assert "创建 skill" in skills["skill-authoring"].manifest.triggers.keywords
    assert "测试失败" in skills["test-failure-fix"].manifest.triggers.keywords
    assert onboarding.requested_tools <= {
        "git_diff",
        "git_status",
        "list_files",
        "read_file",
        "search_text",
    }


def test_generate_native_skill_and_refuse_overwrite_or_unknown_tools(tmp_path: Path) -> None:
    tools = known_tools(tmp_path)
    target = create_skill(
        skill_id="release-review",
        output_root=tmp_path / "drafts",
        name="Release Review",
        summary="Review a release using repository evidence.",
        requested_tools=["list_files", "read_file", "git_diff"],
        before_finish=["git_diff_reviewed"],
        known_tools=tools,
    )

    report = validate_skill(target.parent, known_tools=tools)
    assert report.skill_id == "release-review"
    assert report.requested_tools == ("git_diff", "list_files", "read_file")
    assert "Describe the concrete outcome" in target.read_text(encoding="utf-8")

    with pytest.raises(SkillValidationError, match="Refusing to overwrite"):
        create_skill(
            skill_id="release-review",
            output_root=tmp_path / "drafts",
            known_tools=tools,
        )

    with pytest.raises(SkillValidationError, match="unknown tools"):
        create_skill(
            skill_id="unsafe-draft",
            output_root=tmp_path / "drafts",
            requested_tools=["invented_tool"],
            known_tools=tools,
        )
    assert not (tmp_path / "drafts" / "unsafe-draft").exists()


def test_convert_claude_style_skill_strips_hooks_and_reports_unknown_tools(
    tmp_path: Path,
) -> None:
    source = tmp_path / "foreign" / "SKILL.md"
    source.parent.mkdir()
    source.write_text(
        """---
name: Legacy Debug Helper
description: Diagnose an issue with direct evidence.
allowed-tools: Read, Grep, Bash(git diff:*), WebFetch
hooks:
  on_activate: bootstrap.sh
model: provider-specific-model
---

# Workflow

Read the implementation, locate the failure, and explain the repair.
""",
        encoding="utf-8",
    )

    report = convert_skill(
        source,
        output_root=tmp_path / "converted",
        known_tools=known_tools(tmp_path),
        skill_id="legacy-debug-helper",
    )

    assert report.source_format == "claude"
    assert report.mapped_tools == ("read_file", "run_command", "search_text")
    assert "hooks" in report.dropped_fields
    assert any("WebFetch" in warning for warning in report.warnings)
    assert any("Executable fields were removed" in warning for warning in report.warnings)
    rendered = report.target.read_text(encoding="utf-8")
    assert "bootstrap.sh" not in rendered
    assert "provider-specific-model" not in rendered
    assert "Read the implementation" in rendered
    validate_skill(report.target, known_tools=known_tools(tmp_path))


def test_install_update_and_uninstall_validated_global_skill(tmp_path: Path) -> None:
    tools = known_tools(tmp_path)
    draft = create_skill(
        skill_id="installed-skill",
        output_root=tmp_path / "drafts",
        known_tools=tools,
    )
    global_root = tmp_path / "home" / "skills"

    installed = install_skill(draft, global_root=global_root, known_tools=tools)
    assert installed == global_root / "installed-skill" / "SKILL.md"
    registry = SkillRegistry(system_root=tmp_path / "system", global_root=global_root)
    metadata = registry.metadata("installed-skill")
    assert metadata.source == SkillSource.GLOBAL

    with pytest.raises(SkillValidationError, match="already exists"):
        install_skill(draft, global_root=global_root, known_tools=tools)

    draft.write_text(
        draft.read_text(encoding="utf-8").replace("version: 1.0.0", "version: 1.0.1"),
        encoding="utf-8",
    )
    updated = install_skill(draft, global_root=global_root, known_tools=tools, replace=True)
    assert "version: 1.0.1" in updated.read_text(encoding="utf-8")

    removed = uninstall_skill("installed-skill", global_root=global_root)
    assert removed == global_root / "installed-skill"
    assert not removed.exists()


def test_cli_init_validate_convert_and_builtin_authoring_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REPORIVET_HOME", str(tmp_path / "home"))
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
    drafts = tmp_path / "drafts"

    assert (
        cli(
            [
                "skill",
                "init",
                "generated-skill",
                "--output",
                str(drafts),
                "--tool",
                "read_file",
            ],
            console=console,
        )
        == 0
    )
    generated = drafts / "generated-skill" / "SKILL.md"
    assert cli(["skill", "validate", str(generated)], console=console) == 0
    assert "Skill valid: generated-skill@1.0.0" in buffer.getvalue()
    assert cli(["skill", "install", str(generated)], console=console) == 0
    assert (tmp_path / "home" / "skills" / "generated-skill" / "SKILL.md").is_file()

    plain = tmp_path / "plain" / "SKILL.md"
    plain.parent.mkdir()
    plain.write_text("# Plain workflow\n\nInspect and report.", encoding="utf-8")
    assert (
        cli(
            [
                "skill",
                "convert",
                str(plain),
                "--id",
                "plain-converted",
                "--output",
                str(drafts),
            ],
            console=console,
        )
        == 0
    )
    assert "Detected format: generic-markdown" in buffer.getvalue()

    builtin = (
        Path(__file__).parents[1]
        / "src"
        / "repo_rivet"
        / "builtin_skills"
        / "skill-authoring"
        / "SKILL.md"
    )
    metadata = load_metadata(builtin, SkillSource.SYSTEM)
    assert metadata.manifest.id == "skill-authoring"
    validate_skill(builtin, known_tools=known_tools(tmp_path))
