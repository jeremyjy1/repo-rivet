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


def test_packaged_system_skills_use_portable_front_matter(tmp_path: Path) -> None:
    system_root = Path(__file__).parents[1] / "src" / "repo_rivet" / "builtin_skills"
    registry = SkillRegistry(system_root=system_root, global_root=tmp_path / "global-skills")

    skills = {item.manifest.name: load_bundle(item) for item in registry.system_skills()}

    assert set(skills) == {"repository-onboarding", "skill-authoring", "test-failure-fix"}
    for name, bundle in skills.items():
        report = validate_skill(bundle.path)
        assert report.skill_id == name
        assert bundle.manifest.description
        assert bundle.version == "2.0.0"
        rendered = bundle.path.read_text(encoding="utf-8").split("---", 2)[1]
        for private_field in (
            "schema_version:",
            "requested_tools:",
            "requirements:",
            "triggers:",
            "compatible_modes:",
        ):
            assert private_field not in rendered


def test_generate_portable_skill_and_refuse_overwrite_or_invalid_name(tmp_path: Path) -> None:
    target = create_skill(
        skill_id="release-review",
        output_root=tmp_path / "drafts",
        description="Review a release. Use when release evidence must be checked.",
    )

    report = validate_skill(target.parent)
    assert report.skill_id == "release-review"
    rendered = target.read_text(encoding="utf-8")
    assert "name: release-review" in rendered
    assert "description:" in rendered
    assert "requested_tools" not in rendered

    with pytest.raises(SkillValidationError, match="Refusing to overwrite"):
        create_skill(skill_id="release-review", output_root=tmp_path / "drafts")
    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        create_skill(skill_id="Invalid Name", output_root=tmp_path / "drafts")


def test_convert_skill_removes_host_metadata_and_preserves_resources(tmp_path: Path) -> None:
    source = tmp_path / "foreign" / "SKILL.md"
    reference = source.parent / "references" / "details.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("# Details\n", encoding="utf-8")
    source.write_text(
        """---
name: Legacy Debug Helper
description: Diagnose an issue with direct evidence. Use for debugging requests.
allowed-tools: Read Grep Bash
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
        skill_id="legacy-debug-helper",
    )

    assert report.source_format == "claude"
    assert set(report.dropped_fields) == {"hooks", "model"}
    assert any("Executable metadata was removed" in warning for warning in report.warnings)
    rendered = report.target.read_text(encoding="utf-8")
    assert "bootstrap.sh" not in rendered
    assert "provider-specific-model" not in rendered
    assert "allowed-tools:" in rendered
    assert (report.target.parent / "references" / "details.md").is_file()
    validate_skill(report.target)


def test_install_replace_and_uninstall_complete_skill_package(tmp_path: Path) -> None:
    draft = create_skill(skill_id="installed-skill", output_root=tmp_path / "drafts")
    asset = draft.parent / "assets" / "template.txt"
    asset.parent.mkdir()
    asset.write_text("template", encoding="utf-8")
    global_root = tmp_path / "home" / "skills"

    installed = install_skill(draft.parent, global_root=global_root)
    assert installed == global_root / "installed-skill" / "SKILL.md"
    assert (installed.parent / "assets" / "template.txt").read_text() == "template"
    registry = SkillRegistry(system_root=tmp_path / "system", global_root=global_root)
    assert registry.metadata("global:installed-skill").source == SkillSource.GLOBAL

    with pytest.raises(SkillValidationError, match="already exists"):
        install_skill(draft, global_root=global_root)

    draft.write_text(
        draft.read_text(encoding="utf-8").replace("version: 1.0.0", "version: 1.0.1"),
        encoding="utf-8",
    )
    updated = install_skill(draft.parent, global_root=global_root, replace=True)
    assert "version: 1.0.1" in updated.read_text(encoding="utf-8")

    removed = uninstall_skill("installed-skill", global_root=global_root)
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
                "--description",
                "Generate guidance. Use for generated skill tasks.",
                "--output",
                str(drafts),
            ],
            console=console,
        )
        == 0
    )
    generated = drafts / "generated-skill" / "SKILL.md"
    assert cli(["skill", "validate", str(generated.parent)], console=console) == 0
    assert "Skill valid: generated-skill@1.0.0" in buffer.getvalue()
    assert cli(["skill", "install", str(generated.parent)], console=console) == 0

    plain = tmp_path / "plain" / "SKILL.md"
    plain.parent.mkdir()
    plain.write_text("# Plain workflow\n\nInspect and report.", encoding="utf-8")
    assert (
        cli(
            [
                "skill",
                "convert",
                str(plain),
                "--name",
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
    assert metadata.manifest.name == "skill-authoring"
    validate_skill(builtin)
