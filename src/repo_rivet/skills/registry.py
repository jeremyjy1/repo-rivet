"""Deterministic discovery for packaged system and user-global Skills."""

from __future__ import annotations

from pathlib import Path

from repo_rivet.skills.errors import SkillNotFoundError, SkillValidationError
from repo_rivet.skills.loader import load_bundle, load_metadata
from repo_rivet.skills.models import SkillBundle, SkillMetadata, SkillSource


class SkillRegistry:
    def __init__(self, *, system_root: Path, global_root: Path) -> None:
        self.roots = (
            (SkillSource.SYSTEM, system_root),
            (SkillSource.GLOBAL, global_root),
        )
        self._index: dict[str, SkillMetadata] | None = None
        self._errors: dict[str, str] = {}

    def discover(self, *, refresh: bool = False) -> tuple[SkillMetadata, ...]:
        if self._index is not None and not refresh:
            return tuple(self._index[key] for key in sorted(self._index))
        index: dict[str, SkillMetadata] = {}
        errors: dict[str, str] = {}
        for source, root in self.roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*/SKILL.md")):
                key = f"{source.value}:{path.parent.name}"
                try:
                    metadata = load_metadata(path, source)
                except SkillValidationError as error:
                    errors[key] = str(error)
                    continue
                if metadata.qualified_id in index:
                    errors[key] = f"Duplicate Skill {metadata.qualified_id!r}: {path}"
                    continue
                index[metadata.qualified_id] = metadata
        self._index = index
        self._errors = errors
        return tuple(index[key] for key in sorted(index))

    def _resolve_id(self, skill_id: str) -> str:
        self.discover()
        assert self._index is not None
        if skill_id in self._index or skill_id in self._errors:
            return skill_id
        if ":" in skill_id:
            return skill_id
        matches = [key for key, item in self._index.items() if item.manifest.name == skill_id]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SkillValidationError(
                f"Skill name {skill_id!r} is ambiguous; use one of: " + ", ".join(sorted(matches))
            )
        error_matches = [key for key in self._errors if key.rsplit(":", 1)[-1] == skill_id]
        if len(error_matches) == 1:
            return error_matches[0]
        return skill_id

    def metadata(self, skill_id: str) -> SkillMetadata:
        resolved = self._resolve_id(skill_id)
        assert self._index is not None
        if resolved not in self._index and resolved not in self._errors:
            self.discover(refresh=True)
            resolved = self._resolve_id(skill_id)
            assert self._index is not None
        if resolved in self._errors:
            raise SkillValidationError(self._errors[resolved])
        try:
            return self._index[resolved]
        except KeyError:
            available = ", ".join(sorted(self._index)) or "none"
            raise SkillNotFoundError(
                f"Unknown skill {skill_id!r}. Available skills: {available}"
            ) from None

    def load(self, skill_id: str) -> SkillBundle:
        metadata = self.metadata(skill_id)
        try:
            return load_bundle(metadata)
        except SkillValidationError as error:
            if "metadata changed during activation" not in str(error):
                raise
        self.discover(refresh=True)
        return load_bundle(self.metadata(metadata.qualified_id))

    def system_skills(self) -> tuple[SkillMetadata, ...]:
        return tuple(item for item in self.discover() if item.source == SkillSource.SYSTEM)

    def global_skills(self) -> tuple[SkillMetadata, ...]:
        return tuple(item for item in self.discover() if item.source == SkillSource.GLOBAL)

    def discovery_errors(self) -> tuple[tuple[str, str], ...]:
        self.discover()
        return tuple(sorted(self._errors.items()))
