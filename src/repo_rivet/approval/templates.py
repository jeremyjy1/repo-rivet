"""Deterministic approval templates over unified semantic facts."""

from dataclasses import dataclass

from repo_rivet.approval.models import (
    AnalysisLevel,
    ApprovalRequest,
    ArtifactProvenance,
    EffectScope,
    ExecutableOrigin,
    OperationClass,
)


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    name: str
    reason: str
    constraints: list[str]


class DeterministicApprovalTemplates:
    """Match narrowly bounded operations without relying on a risk-number threshold."""

    def match(self, request: ApprovalRequest) -> TemplateMatch | None:
        facts = request.facts
        if facts.analysis_level == AnalysisLevel.OPAQUE:
            return None
        if facts.accesses_network or facts.requires_privilege or facts.outside_workspace:
            return None
        if facts.touches_sensitive_paths or facts.delete_paths:
            return None

        match = self._bounded_build(request)
        if match is not None:
            return match
        match = self._bounded_static_check(request)
        if match is not None:
            return match
        match = self._bounded_test(request)
        if match is not None:
            return match
        match = self._session_artifact_run(request)
        if match is not None:
            return match
        return self._managed_generation(request)

    @staticmethod
    def _bounded_build(request: ApprovalRequest) -> TemplateMatch | None:
        facts = request.facts
        if facts.operation_class != OperationClass.BUILD:
            return None
        if facts.analysis_level != AnalysisLevel.EXACT:
            return None
        if facts.executable_origin != ExecutableOrigin.TRUSTED_TOOLCHAIN:
            return None
        if not facts.read_paths or not facts.write_paths:
            return None
        if facts.effect_scope not in {EffectScope.MANAGED, EffectScope.WORKSPACE}:
            return None
        allowed_provenance = {
            ArtifactProvenance.NEW,
            ArtifactProvenance.SESSION_GENERATED,
            ArtifactProvenance.STALE,
        }
        if not set(facts.output_provenance.values()) <= allowed_provenance:
            return None
        return TemplateMatch(
            name="bounded_build",
            reason=(
                "exact trusted-toolchain build has explicit workspace inputs and only "
                "creates or refreshes session-owned outputs"
            ),
            constraints=sorted(facts.constraints),
        )

    @staticmethod
    def _bounded_static_check(request: ApprovalRequest) -> TemplateMatch | None:
        facts = request.facts
        if facts.operation_class != OperationClass.STATIC_CHECK:
            return None
        if facts.analysis_level != AnalysisLevel.EXACT:
            return None
        if facts.executable_origin != ExecutableOrigin.TRUSTED_TOOLCHAIN:
            return None
        if facts.write_paths or "filesystem_write" in facts.explicit_effects:
            return None
        return TemplateMatch(
            name="bounded_static_check",
            reason="exact trusted static-analysis command has no declared write effect",
            constraints=sorted(facts.constraints),
        )

    @staticmethod
    def _bounded_test(request: ApprovalRequest) -> TemplateMatch | None:
        facts = request.facts
        if request.tool_name != "run_verification":
            return None
        if facts.operation_class != OperationClass.TEST or facts.verification_kind is None:
            return None
        if facts.executable_origin != ExecutableOrigin.TRUSTED_TOOLCHAIN:
            return None
        if facts.analysis_level not in {AnalysisLevel.EXACT, AnalysisLevel.EXPANDED}:
            return None
        return TemplateMatch(
            name="bounded_test",
            reason="registered verification uses a recognized test runner under local limits",
            constraints=sorted(facts.constraints),
        )

    @staticmethod
    def _session_artifact_run(request: ApprovalRequest) -> TemplateMatch | None:
        facts = request.facts
        if facts.operation_class != OperationClass.RUN_ARTIFACT:
            return None
        if facts.executable_origin != ExecutableOrigin.SESSION_GENERATED:
            return None
        if facts.analysis_level != AnalysisLevel.EXACT:
            return None
        return TemplateMatch(
            name="session_artifact_run",
            reason="executable is an unchanged artifact produced at the current workspace revision",
            constraints=sorted(facts.constraints),
        )

    @staticmethod
    def _managed_generation(request: ApprovalRequest) -> TemplateMatch | None:
        facts = request.facts
        if facts.operation_class != OperationClass.GENERATE:
            return None
        if facts.analysis_level not in {AnalysisLevel.EXACT, AnalysisLevel.EXPANDED}:
            return None
        if facts.executable_origin != ExecutableOrigin.TRUSTED_TOOLCHAIN:
            return None
        if facts.effect_scope != EffectScope.MANAGED:
            return None
        if not facts.read_paths or not facts.write_paths:
            return None
        return TemplateMatch(
            name="managed_generation",
            reason="recognized generator is confined to managed output directories",
            constraints=sorted(facts.constraints),
        )
