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
                try:
                    metadata = load_metadata(path, source)
                except SkillValidationError as error:
                    errors[path.parent.name] = str(error)
                    continue
                skill_id = metadata.manifest.id
                if skill_id in errors:
                    continue
                if skill_id in index:
                    existing = index[skill_id]
                    if existing.source == SkillSource.SYSTEM and source == SkillSource.GLOBAL:
                        errors[f"global:{skill_id}"] = (
                            f"Global Skill id {skill_id!r} conflicts with system Skill "
                            f"{existing.path}: {path}"
                        )
                        continue
                    errors[skill_id] = f"Duplicate skill id {skill_id!r}: {existing.path}, {path}"
                    index.pop(skill_id)
                    continue
                index[skill_id] = metadata
        self._index = index
        self._errors = errors
        return tuple(index[key] for key in sorted(index))

    def metadata(self, skill_id: str) -> SkillMetadata:
        self.discover()
        assert self._index is not None
        if skill_id not in self._index and skill_id not in self._errors:
            self.discover(refresh=True)
            assert self._index is not None
        if skill_id in self._errors:
            raise SkillValidationError(self._errors[skill_id])
        try:
            return self._index[skill_id]
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
        return load_bundle(self.metadata(skill_id))

    def system_skills(self) -> tuple[SkillMetadata, ...]:
        """Return every packaged system Skill in deterministic order."""
        return tuple(item for item in self.discover() if item.source == SkillSource.SYSTEM)

    def global_skills(self) -> tuple[SkillMetadata, ...]:
        """Return user-installed Skills shared by all workspaces."""
        return tuple(item for item in self.discover() if item.source == SkillSource.GLOBAL)

    def discovery_errors(self) -> tuple[tuple[str, str], ...]:
        """Expose invalid or conflicting entries without hiding valid system Skills."""
        self.discover()
        return tuple(sorted(self._errors.items()))
