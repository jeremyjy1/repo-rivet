"""Deterministic approval templates over unified semantic facts."""

from dataclasses import dataclass

from repo_rivet.approval.models import (
    AnalysisLevel,
    ApprovalRequest,
    ArtifactProvenance,
    EffectScope,
    ExecutableOrigin,
    OperationClass,
    RiskLevel,
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

        match = self._reporivet_skill_cli(request)
        if match is not None:
            return match
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

    def match_llm_auto(self, request: ApprovalRequest) -> TemplateMatch | None:
        """Match local writes that are safe to resolve before consulting the LLM reviewer."""
        facts = request.facts
        if request.tool_name not in {"write_file", "edit_file"}:
            return None
        if request.assessment.level > RiskLevel.MEDIUM:
            return None
        if facts.operation_class != OperationClass.EDIT:
            return None
        if facts.analysis_level != AnalysisLevel.EXACT:
            return None
        if facts.effect_scope != EffectScope.WORKSPACE or len(facts.write_paths) != 1:
            return None
        if facts.accesses_network or facts.requires_privilege or facts.outside_workspace:
            return None
        if facts.touches_sensitive_paths or facts.delete_paths or facts.potential_capabilities:
            return None
        if not facts.explicit_effects <= {"filesystem_read", "filesystem_write"}:
            return None
        required_constraints = {"typed_tool", "workspace_path_policy", "snapshot_precondition"}
        if not required_constraints <= facts.constraints:
            return None

        arguments = request.normalized_arguments
        if request.tool_name == "write_file":
            content = arguments.get("content")
            if facts.overwrites_existing or not isinstance(content, dict):
                return None
            if not isinstance(content.get("characters"), int) or not isinstance(
                content.get("sha256"), str
            ):
                return None
            return TemplateMatch(
                name="bounded_workspace_create",
                reason=("typed file creation is confined to one new non-sensitive workspace path"),
                constraints=sorted(facts.constraints),
            )

        if not facts.overwrites_existing:
            return None
        if not isinstance(arguments.get("snapshot_id"), str) or not isinstance(
            arguments.get("prepared_live_hash"), str
        ):
            return None
        operations = arguments.get("operations")
        if not isinstance(operations, list) or not operations:
            return None
        if not isinstance(arguments.get("diff_preview"), str):
            return None
        return TemplateMatch(
            name="bounded_workspace_edit",
            reason=("exact snapshot-bound edit is confined to one non-sensitive workspace file"),
            constraints=sorted(facts.constraints),
        )

    @staticmethod
    def _reporivet_skill_cli(request: ApprovalRequest) -> TemplateMatch | None:
        facts = request.facts
        if "reporivet_skill_cli" not in facts.constraints:
            return None
        if request.tool_name not in {"run_command", "run_verification"}:
            return None
        if facts.analysis_level != AnalysisLevel.EXACT:
            return None
        if facts.executable_origin != ExecutableOrigin.TRUSTED_TOOLCHAIN:
            return None
        if facts.operation_class in {OperationClass.READ, OperationClass.STATIC_CHECK}:
            if facts.write_paths or facts.effect_scope not in {
                EffectScope.NONE,
                EffectScope.WORKSPACE,
            }:
                return None
            return TemplateMatch(
                name="reporivet_skill_inspection",
                reason="trusted RepoRivet CLI performs a bounded Skill read or validation",
                constraints=sorted(facts.constraints),
            )
        if facts.operation_class != OperationClass.GENERATE:
            return None
        if not facts.write_paths or facts.effect_scope != EffectScope.WORKSPACE:
            return None
        if set(facts.output_provenance.values()) != {ArtifactProvenance.NEW}:
            return None
        return TemplateMatch(
            name="reporivet_skill_generation",
            reason="trusted RepoRivet CLI creates one new Skill draft inside the workspace",
            constraints=sorted(facts.constraints),
        )

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
