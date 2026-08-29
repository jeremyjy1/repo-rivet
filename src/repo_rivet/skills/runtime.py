"""Session activation, capability narrowing, and pinned-content validation."""

from __future__ import annotations

from repo_rivet.memory.models import MemoryState
from repo_rivet.planning.models import PlanStatus, WorkflowMode
from repo_rivet.skills.errors import SkillError, SkillStaleError, SkillValidationError
from repo_rivet.skills.models import ActiveSkillPin, SkillActivation, SkillBundle, SkillSource
from repo_rivet.skills.registry import SkillRegistry

CONTROL_TOOLS = frozenset(
    {"record_decision", "register_verification", "submit_plan", "update_plan"}
)
# These are Controller protocol tools rather than task capabilities. Keeping the mode-valid
# subset prevents a narrowed Skill from deadlocking decision, verification, or Plan workflows.


class SkillRuntime:
    def __init__(self, registry: SkillRegistry, *, known_tools: set[str]) -> None:
        self.registry = registry
        self.known_tools = known_tools
        system: list[SkillBundle] = []
        for item in registry.system_skills():
            bundle = self._validate_bundle(registry.load(item.manifest.id))
            if not bundle.manifest.activation.automatic:
                raise SkillValidationError(
                    f"System Skill {bundle.manifest.id} must declare automatic activation"
                )
            system.append(bundle)
        self._system = tuple(system)
        self._active: SkillBundle | None = None

    @property
    def system(self) -> tuple[SkillBundle, ...]:
        """Packaged system Skills, eagerly loaded for every runtime."""
        return self._system

    @property
    def active(self) -> SkillBundle | None:
        return self._active

    def sync_system(self, memory: MemoryState) -> tuple[ActiveSkillPin, ...]:
        """Persist the exact packaged Skill set used by this runtime."""
        pins = tuple(
            ActiveSkillPin(
                id=bundle.manifest.id,
                version=bundle.manifest.version,
                content_hash=bundle.content_hash,
                source=SkillSource.SYSTEM,
                activation=SkillActivation.SYSTEM,
            )
            for bundle in self._system
        )
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
        bundle = self._validate_bundle(self.registry.load(skill_id))
        if bundle.source != SkillSource.GLOBAL:
            raise SkillValidationError(
                f"System Skill {skill_id} is already loaded and cannot be selected as the "
                "session's global Skill"
            )
        if not bundle.manifest.activation.explicit:
            raise SkillValidationError(
                f"Global Skill {skill_id} does not allow explicit selection"
            )
        pin = ActiveSkillPin(
            id=bundle.manifest.id,
            version=bundle.manifest.version,
            content_hash=bundle.content_hash,
            source=bundle.source,
            activation=activation,
        )
        changed = memory.active_skill != pin
        memory.active_skill = pin
        self._active = bundle
        if changed:
            memory.skill_completion_recovery_attempts = 0
            self._invalidate_plan(memory)
        return bundle

    def restore(self, memory: MemoryState) -> SkillBundle | None:
        self.sync_system(memory)
        pin = memory.active_skill
        if pin is None:
            self._active = None
            return None
        try:
            bundle = self._validate_bundle(self.registry.load(pin.id))
        except SkillError as error:
            self._invalidate_plan(memory)
            raise SkillStaleError(
                f"Pinned skill {pin.id} is unavailable: {error}; select it again or clear it"
            ) from None
        if bundle.source != SkillSource.GLOBAL:
            self._invalidate_plan(memory)
            raise SkillStaleError(
                f"Pinned skill {pin.id} is now a system Skill and is already loaded; clear the "
                "session's global Skill selection"
            )
        if (
            bundle.source != pin.source
            or bundle.manifest.version != pin.version
            or bundle.content_hash != pin.content_hash
        ):
            self._invalidate_plan(memory)
            raise SkillStaleError(
                f"Pinned skill {pin.id} changed after activation; select it again or clear it"
            )
        self._active = bundle
        return bundle

    def _validate_bundle(self, bundle: SkillBundle) -> SkillBundle:
        unknown = bundle.manifest.requested_tools - self.known_tools
        if unknown:
            raise SkillValidationError(
                f"Skill {bundle.manifest.id} requests unknown tools: "
                + ", ".join(sorted(unknown))
            )
        return bundle

    def clear(self, memory: MemoryState) -> None:
        if memory.active_skill is not None:
            self._invalidate_plan(memory)
        memory.active_skill = None
        memory.skill_completion_recovery_attempts = 0
        self._active = None

    def supports_mode(self, mode: WorkflowMode) -> bool:
        if self._active is None:
            return True
        mode_name = "plan" if mode == WorkflowMode.PLANNING else "execute"
        return mode_name in self._active.manifest.compatible_modes

    def allowed_tool_names(self, mode_tools: set[str]) -> set[str]:
        if self._active is None:
            return mode_tools
        return {
            name
            for name in mode_tools
            if name in CONTROL_TOOLS or name in self._active.manifest.requested_tools
        }

    @staticmethod
    def _invalidate_plan(memory: MemoryState) -> None:
        if memory.plan_artifact is None or memory.plan_artifact.status in {
            PlanStatus.CANCELLED,
            PlanStatus.COMPLETED,
        }:
            return
        memory.plan_artifact.status = PlanStatus.STALE
        memory.workflow_mode = WorkflowMode.PLAN_READY
