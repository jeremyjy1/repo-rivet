"""Models for portable Agent Skills and durable session pins."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SkillSource(StrEnum):
    SYSTEM = "system"
    GLOBAL = "global"


class SkillActivation(StrEnum):
    SYSTEM = "system"
    EXPLICIT = "explicit"
    CONFIG_DEFAULT = "config_default"
    SESSION_RESTORE = "session_restore"


class SkillManifest(BaseModel):
    """Portable ``SKILL.md`` front matter.

    Unknown fields are retained but never interpreted as RepoRivet policy. This keeps discovery
    forward compatible while ensuring a Skill cannot grant itself capabilities.
    """

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )
    description: str = Field(min_length=1, max_length=1_024)
    license: str | None = Field(default=None, max_length=500)
    compatibility: str | None = Field(default=None, max_length=500)
    metadata: dict[str, str] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list, alias="allowed-tools", max_length=100)

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metadata must be a mapping of string keys to string values")
        return {str(key): str(item) for key, item in value.items()}

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def normalize_allowed_tools(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part for part in value.split() if part]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        raise ValueError("allowed-tools must be a string or list")

    @property
    def version(self) -> str | None:
        return self.metadata.get("version")


class SkillMetadata(BaseModel):
    """Lightweight descriptor; the Markdown instruction body is intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    qualified_id: str
    manifest: SkillManifest
    source: SkillSource
    path: Path
    manifest_hash: str
    body_size_bytes: int = Field(ge=0)

    @property
    def version(self) -> str | None:
        return self.manifest.version


class SkillBundle(SkillMetadata):
    """A selected Skill with lazily loaded instructions and resource inventory."""

    body: str
    content_hash: str
    estimated_prompt_tokens: int = Field(ge=0)
    resource_files: tuple[str, ...] = ()
    script_files: tuple[str, ...] = ()
    asset_files: tuple[str, ...] = ()


class ActiveSkillPin(BaseModel):
    """Durable identity of the exact Skill instructions selected for a session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    version: str | None = None
    content_hash: str
    source: SkillSource
    activation: SkillActivation
