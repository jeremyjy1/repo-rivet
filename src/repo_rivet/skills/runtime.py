"""Session activation, lazy routing, and content pinning for portable Skills."""

from __future__ import annotations

import re

from repo_rivet.memory.models import MemoryState
from repo_rivet.planning.models import PlanStatus, WorkflowMode
from repo_rivet.skills.errors import SkillError, SkillStaleError, SkillValidationError
from repo_rivet.skills.models import (
    ActiveSkillPin,
    SkillActivation,
    SkillBundle,
    SkillMetadata,
    SkillSource,
)
from repo_rivet.skills.registry import SkillRegistry

_ROUTING_STOP_WORDS = {
    "agent",
    "and",
    "for",
    "from",
    "tasks",
    "that",
    "the",
    "this",
    "use",
    "when",
    "with",
}


class SkillRuntime:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry
        self._system = registry.system_skills()
        self._system_cache: dict[str, SkillBundle] = {}
        self._system_failures: dict[str, str] = {}
        self._active: SkillBundle | None = None
        self._automatic_load_errors: list[tuple[str, str]] = []

    @property
    def system(self) -> tuple[SkillMetadata, ...]:
        """Indexed system descriptors; their instruction bodies remain unloaded."""
        return self._system

    @property
    def active(self) -> SkillBundle | None:
        return self._active

    @property
    def automatic_load_errors(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._automatic_load_errors)

    @staticmethod
    def _pin(metadata: SkillMetadata, activation: SkillActivation) -> ActiveSkillPin:
        return ActiveSkillPin(
            id=metadata.qualified_id,
            name=metadata.manifest.name,
            version=metadata.version,
            content_hash=metadata.manifest_hash,
            source=metadata.source,
            activation=activation,
        )

    def sync_system(self, memory: MemoryState) -> tuple[ActiveSkillPin, ...]:
        pins = tuple(self._pin(item, SkillActivation.SYSTEM) for item in self._system)
        previous = tuple(memory.system_skills)
        memory.system_skills = list(pins)
        if previous and previous != pins:
            self._invalidate_plan(memory)
        return pins

    def activate(
        self,
        memory: MemoryState,
        skill_id: str,
        *,
        activation: SkillActivation = SkillActivation.EXPLICIT,
    ) -> SkillBundle:
        bundle = self.registry.load(skill_id)
        if bundle.source != SkillSource.GLOBAL:
            raise SkillValidationError(
                f"System Skill {bundle.qualified_id} is automatically routed and cannot be "
                "selected as the session's global Skill"
            )
        pin = self._pin(bundle, activation)
        changed = memory.active_skill != pin
        memory.active_skill = pin
        self._active = bundle
        if changed:
            self._invalidate_plan(memory)
        return bundle

    def restore(self, memory: MemoryState) -> SkillBundle | None:
        self.sync_system(memory)
        pin = memory.active_skill
        if pin is None:
            self._active = None
            return None
        try:
            bundle = self.registry.load(pin.id)
        except SkillError as error:
            self._invalidate_plan(memory)
            raise SkillStaleError(
                f"Pinned skill {pin.id} is unavailable: {error}; select it again or clear it"
            ) from None
        if bundle.source != SkillSource.GLOBAL:
            self._invalidate_plan(memory)
            raise SkillStaleError(f"Pinned skill {pin.id} is no longer a global Skill")
        if self._pin(bundle, pin.activation) != pin:
            self._invalidate_plan(memory)
            raise SkillStaleError(
                f"Pinned skill {pin.id} changed after activation; select it again or clear it"
            )
        self._active = bundle
        return bundle

    def clear(self, memory: MemoryState) -> None:
        if memory.active_skill is not None:
            self._invalidate_plan(memory)
        memory.active_skill = None
        self._active = None

    def system_for_task(self, task: str) -> tuple[SkillBundle, ...]:
        """Match standard descriptions, then lazily load only selected instructions."""
        bundles: list[SkillBundle] = []
        self._automatic_load_errors.clear()
        for metadata in self._system:
            if not _description_matches(metadata.manifest.description, task):
                continue
            if metadata.qualified_id in self._system_cache:
                bundles.append(self._system_cache[metadata.qualified_id])
                continue
            if metadata.qualified_id in self._system_failures:
                continue
            try:
                bundle = self.registry.load(metadata.qualified_id)
                self._system_cache[metadata.qualified_id] = bundle
                bundles.append(bundle)
            except SkillError as error:
                rendered = str(error)
                self._system_failures[metadata.qualified_id] = rendered
                self._automatic_load_errors.append((metadata.qualified_id, rendered))
        return tuple(bundles)

    @staticmethod
    def _invalidate_plan(memory: MemoryState) -> None:
        if memory.plan_artifact is None or memory.plan_artifact.status in {
            PlanStatus.CANCELLED,
            PlanStatus.COMPLETED,
        }:
            return
        memory.plan_artifact.status = PlanStatus.STALE
        memory.workflow_mode = WorkflowMode.PLAN_READY


def _description_matches(description: str, task: str) -> bool:
    normalized_task = " ".join(task.casefold().replace("_", " ").replace("-", " ").split())
    english_terms = {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9+.#-]{2,}", description.casefold())
        if term not in _ROUTING_STOP_WORDS
    }
    if sum(term in normalized_task for term in english_terms) >= 2:
        return True
    # Descriptions may include comma-separated CJK routing terms. Keeping those terms explicit in
    # the portable description makes routing useful without a private trigger schema.
    cjk_terms = re.findall(r"[\u3400-\u9fff]{2,8}", description)
    return any(term in task for term in cjk_terms)
