"""Strict serializable models for declarative RepoRivet skills."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SkillSource(StrEnum):
    SYSTEM = "system"
    GLOBAL = "global"


class SkillActivation(StrEnum):
    SYSTEM = "system"
    EXPLICIT = "explicit"
    CONFIG_DEFAULT = "config_default"
    SESSION_RESTORE = "session_restore"


class SkillActivationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    explicit: bool = True
    automatic: bool = False


class SkillTriggers(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_types: list[str] = Field(default_factory=list, max_length=50)
    file_globs: list[str] = Field(default_factory=list, max_length=50)
    project_markers: list[str] = Field(default_factory=list, max_length=50)
    keywords: list[str] = Field(default_factory=list, max_length=50)


class SkillRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    before_edit: list[str] = Field(default_factory=list, max_length=20)
    before_finish: list[str] = Field(default_factory=list, max_length=20)


class SkillLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_prompt_tokens: int = Field(default=2_000, ge=100, le=20_000)
    max_active_support_skills: int = Field(default=0, ge=0, le=10)


class SkillManifest(BaseModel):
    """Validated YAML front matter; it may request but never grant capabilities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1, le=1)
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    name: str = Field(min_length=1, max_length=100)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    summary: str = Field(min_length=1, max_length=500)
    category: str = Field(default="workflow", min_length=1, max_length=50)
    activation: SkillActivationPolicy = Field(default_factory=SkillActivationPolicy)
    triggers: SkillTriggers = Field(default_factory=SkillTriggers)
    compatible_modes: set[str] = Field(default_factory=lambda: {"plan", "execute"})
    requested_tools: set[str] = Field(min_length=1, max_length=50)
    requirements: SkillRequirements = Field(default_factory=SkillRequirements)
    verification_profiles: list[str] = Field(default_factory=list, max_length=20)
    limits: SkillLimits = Field(default_factory=SkillLimits)

    @model_validator(mode="after")
    def validate_modes_and_requirements(self) -> SkillManifest:
        unknown_modes = self.compatible_modes - {"plan", "execute"}
        if unknown_modes:
            raise ValueError("unknown compatible modes: " + ", ".join(sorted(unknown_modes)))
        if not self.compatible_modes:
            raise ValueError("compatible_modes must not be empty")
        known_requirements = {
            "target_snapshot_current",
            "target_range_seen",
            "plan_approved",
            "no_active_processes",
            "required_build_passed",
            "required_tests_passed",
            "required_behavior_checks_passed",
            "no_stale_verification",
            "git_diff_reviewed",
        }
        declared = set(self.requirements.before_edit) | set(self.requirements.before_finish)
        unknown = declared - known_requirements
        if unknown:
            raise ValueError("unknown requirements: " + ", ".join(sorted(unknown)))
        return self


class SkillMetadata(BaseModel):
    """Lightweight discovery record; the Markdown body is intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    manifest: SkillManifest
    source: SkillSource
    path: Path


class SkillBundle(SkillMetadata):
    """Fully loaded system or selected-global Skill."""

    body: str
    content_hash: str
    estimated_prompt_tokens: int = Field(ge=0)


class ActiveSkillPin(BaseModel):
    """Durable identity of the exact global Skill selected for a session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    version: str
    content_hash: str
    source: SkillSource
    activation: SkillActivation
