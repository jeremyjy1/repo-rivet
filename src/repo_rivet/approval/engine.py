"""Layer deterministic safety, modes, grants, LLM advice, and human decisions."""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from repo_rivet.approval.grant_store import ApprovalGrantStore
from repo_rivet.approval.hard_policy import HardSafetyPolicy
from repo_rivet.approval.human_approver import HumanApprover
from repo_rivet.approval.llm_reviewer import LLMApprovalReviewer
from repo_rivet.approval.models import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalMode,
    ApprovalRequest,
    ApprovalScope,
    Capability,
    LLMReviewResult,
    OperationClass,
    RiskLevel,
)
from repo_rivet.approval.normalizer import RequestNormalizer
from repo_rivet.approval.review_context import (
    AUTO_APPROVAL_BLOCKING_EFFECTS,
    IMPORTANT_EFFECTS,
)
from repo_rivet.approval.risk_analyzer import RiskAnalyzer
from repo_rivet.approval.semantic_analyzer import artifact_key
from repo_rivet.approval.templates import DeterministicApprovalTemplates
from repo_rivet.memory.models import ArtifactRecord


class EventSink(Protocol):
    def log(self, event_type: str, **data: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class ApprovalOutcome:
    request: ApprovalRequest
    decision: ApprovalDecision


class ApprovalEngine:
    """Authorize complete tool requests and detect approval-to-execution drift."""

    def __init__(
        self,
        *,
        mode: ApprovalMode,
        normalizer: RequestNormalizer,
        risk_analyzer: RiskAnalyzer,
        hard_policy: HardSafetyPolicy,
        grant_store: ApprovalGrantStore,
        human_approver: HumanApprover,
        llm_reviewer: LLMApprovalReviewer | None = None,
        max_llm_risk: RiskLevel = RiskLevel.MEDIUM,
        event_logger: EventSink | None = None,
        approval_templates: DeterministicApprovalTemplates | None = None,
    ) -> None:
        self.mode = mode
        self.normalizer = normalizer
        self.risk_analyzer = risk_analyzer
        self.hard_policy = hard_policy
        self.grant_store = grant_store
        self.human_approver = human_approver
        self.llm_reviewer = llm_reviewer
        self.max_llm_risk = max_llm_risk
        self.event_logger = event_logger
        self.approval_templates = approval_templates or DeterministicApprovalTemplates()
        self.risk_analyzer.bind(self.grant_store.memory)
        self.sync_memory_rule()

    @property
    def session_id(self) -> str:
        return self.grant_store.memory.session_id

    def set_mode(self, mode: ApprovalMode) -> None:
        """Switch policy immediately and persist the explicit session override."""
        previous_mode = self.mode
        self.mode = mode
        self.grant_store.memory.approval_mode_override = mode
        if self.grant_store.memory.runtime is not None and previous_mode != mode:
            self.grant_store.memory.runtime.revisions.approval_policy += 1
        self.grant_store.memory.denied_request_fingerprints.clear()
        self.grant_store.memory.approval_denial_guidance.clear()
        self.sync_memory_rule()
        self._log(
            "approval_mode_changed",
            previous_mode=previous_mode.value,
            mode=mode.value,
        )

    def sync_memory_rule(self) -> None:
        """Keep the model's fixed safety context aligned with the active mode."""
        fixed = self.grant_store.memory.fixed
        if fixed is None:
            return
        prefix = "Current approval mode:"
        fixed.safety_rules[:] = [rule for rule in fixed.safety_rules if not rule.startswith(prefix)]
        fixed.safety_rules.append(f"Current approval mode: {self.mode.value}.")

    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        capabilities: set[Capability] | frozenset[Capability],
        session_id: str,
    ) -> ApprovalOutcome:
        request = self.normalizer.normalize(
            tool_name=tool_name,
            arguments=arguments,
            session_id=session_id,
            declared_capabilities=capabilities,
        )
        request.task_summary = self._task_summary()
        self.risk_analyzer.assess(request)
        self._log_request(request)

        decision = self.hard_policy.evaluate(request)
        if decision is None:
            grant = self.grant_store.match(request)
            if grant is not None:
                action = ApprovalAction(grant.action)
                decision = self._decision(
                    request,
                    action=action,
                    source="session_grant",
                    reason=f"matched an exact session {grant.action} decision",
                    scope=ApprovalScope.SESSION_EXACT,
                    guidance=grant.guidance,
                )
            elif request.fingerprint in self.grant_store.memory.denied_request_fingerprints:
                decision = self._decision(
                    request,
                    action=ApprovalAction.DENY,
                    source="prior_denial",
                    reason="the exact request was already denied during this task",
                    guidance=self.grant_store.memory.approval_denial_guidance.get(
                        request.fingerprint
                    ),
                )
        if decision is None:
            decision = self._decide_by_mode(request)

        if decision.scope == ApprovalScope.SESSION_EXACT:
            self.grant_store.remember(
                request,
                decision.action,
                guidance=decision.guidance,
            )
        elif decision.action == ApprovalAction.DENY:
            self.grant_store.memory.denied_request_fingerprints.add(request.fingerprint)
        if decision.action == ApprovalAction.DENY and decision.guidance:
            self.grant_store.memory.approval_denial_guidance[request.fingerprint] = (
                decision.guidance
            )
        self._log_decision(request, decision)
        return ApprovalOutcome(request=request, decision=decision)

    def revalidate(self, outcome: ApprovalOutcome) -> ApprovalDecision | None:
        refreshed = self.normalizer.refresh(outcome.request)
        refreshed.task_summary = outcome.request.task_summary
        self.risk_analyzer.assess(refreshed)
        if refreshed.fingerprint != outcome.decision.request_fingerprint:
            decision = self._decision(
                refreshed,
                action=ApprovalAction.DENY,
                source="execution_revalidation",
                reason="request path or normalized arguments changed after approval",
            )
            self._log_decision(refreshed, decision)
            return decision
        if refreshed.facts != outcome.request.facts:
            decision = self._decision(
                refreshed,
                action=ApprovalAction.DENY,
                source="execution_revalidation",
                reason=(
                    "executable identity, effect scope, or artifact provenance changed "
                    "after approval"
                ),
            )
            self._log_decision(refreshed, decision)
            return decision
        hard_decision = self.hard_policy.evaluate(refreshed)
        if hard_decision is not None:
            self._log_decision(refreshed, hard_decision)
            return hard_decision
        return None

    def record_execution(self, outcome: ApprovalOutcome, *, ok: bool, metadata: Any) -> None:
        if ok:
            self._record_artifacts(outcome)
        self._log(
            "approved_tool_executed",
            request_id=outcome.request.request_id,
            tool=outcome.request.tool_name,
            fingerprint=outcome.request.fingerprint,
            ok=ok,
            metadata=metadata,
        )

    def record_execution_started(self, outcome: ApprovalOutcome) -> None:
        """Record the point after revalidation when local execution actually begins."""
        self._log(
            "approved_tool_started",
            request_id=outcome.request.request_id,
            tool=outcome.request.tool_name,
            fingerprint=outcome.request.fingerprint,
        )

    def _decide_by_mode(self, request: ApprovalRequest) -> ApprovalDecision:
        if self.mode == ApprovalMode.ALLOW_ALL:
            return self._decision(
                request,
                action=ApprovalAction.ALLOW,
                source="allow_all_mode",
                reason="allowed by allow-all mode after hard-safety checks",
            )
        if self.mode == ApprovalMode.ALWAYS_ASK:
            return self.human_approver.ask(request)
        if request.assessment.obviously_safe:
            return self._decision(
                request,
                action=ApprovalAction.ALLOW,
                source="safe_rule",
                reason="matched a narrow deterministic harmless-tool rule",
            )
        template = self.approval_templates.match(request)
        if template is not None:
            return self._decision(
                request,
                action=ApprovalAction.ALLOW,
                source=f"semantic_template:{template.name}",
                reason=template.reason,
                constraints=template.constraints,
            )
        if self.mode == ApprovalMode.SAFE_AUTO:
            return self.human_approver.ask(request)
        llm_auto_template = self.approval_templates.match_llm_auto(request)
        if llm_auto_template is not None:
            return self._decision(
                request,
                action=ApprovalAction.ALLOW,
                source=f"llm_auto_template:{llm_auto_template.name}",
                reason=llm_auto_template.reason,
                constraints=llm_auto_template.constraints,
            )
        return self._review_with_llm_or_human(request)

    def _review_with_llm_or_human(self, request: ApprovalRequest) -> ApprovalDecision:
        if self.llm_reviewer is None:
            return self.human_approver.ask(request)
        self._log(
            "llm_approval_review_started",
            request_id=request.request_id,
            tool=request.tool_name,
            fingerprint=request.fingerprint,
            risk=request.assessment.level.name.lower(),
        )
        review_started = time.monotonic()
        failure: dict[str, Any] | None = None
        try:
            review = self.llm_reviewer.review(request)
        except Exception as error:
            review = None
            failure = {
                "error_type": type(error).__name__,
                "stage": "review",
            }
        if review is None and failure is None:
            value = getattr(self.llm_reviewer, "last_failure", None)
            failure = (
                dict(value)
                if isinstance(value, dict)
                else {
                    "error_type": "UnavailableReview",
                    "stage": "review",
                }
            )
        self._log_review(
            request,
            review,
            duration_seconds=time.monotonic() - review_started,
            failure=failure,
        )
        if review is not None and self._accept_llm_approval(request, review):
            return self._decision(
                request,
                action=ApprovalAction.ALLOW,
                source="llm_reviewer",
                reason=review.reason,
                constraints=review.required_constraints,
            )
        return self.human_approver.ask(request, llm_review=review)

    def _accept_llm_approval(
        self,
        request: ApprovalRequest,
        review: LLMReviewResult,
    ) -> bool:
        if review.recommendation != "allow":
            return False
        if review.task_relevance not in {"required", "helpful"} or review.unknowns:
            return False
        review_risk = RiskLevel[review.risk_level.upper()]
        if request.assessment.level > self.max_llm_risk or review_risk > self.max_llm_risk:
            return False

        deterministic_effects = request.facts.explicit_effects
        deterministic_important = deterministic_effects & IMPORTANT_EFFECTS
        recognized = set(review.recognized_effects)
        if deterministic_effects & AUTO_APPROVAL_BLOCKING_EFFECTS:
            return False
        if not deterministic_important <= recognized:
            return False
        if (recognized & IMPORTANT_EFFECTS) - deterministic_effects:
            return False
        return set(review.required_constraints) <= request.facts.constraints

    def _record_artifacts(self, outcome: ApprovalOutcome) -> None:
        request = outcome.request
        if request.facts.operation_class not in {
            OperationClass.BUILD,
            OperationClass.GENERATE,
        }:
            return
        memory = self.grant_store.memory
        workspace = Path(request.workspace)
        for path_value in request.facts.write_paths:
            path = Path(path_value)
            if not path.is_relative_to(workspace) or not path.is_file():
                continue
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
            memory.artifact_registry[artifact_key(path, workspace)] = ArtifactRecord(
                path=artifact_key(path, workspace),
                artifact_type=(
                    "executable"
                    if request.facts.operation_class == OperationClass.BUILD
                    else "generated"
                ),
                created_by_session=request.session_id,
                created_by_request=request.request_id,
                producer_operation=request.facts.operation_class,
                source_paths=[
                    artifact_key(Path(source), workspace)
                    for source in request.facts.read_paths
                    if Path(source).is_relative_to(workspace)
                ],
                content_sha256=digest,
                workspace_revision=memory.workspace_revision,
            )
            self._log(
                "artifact_registered",
                path=artifact_key(path, workspace),
                producer_operation=request.facts.operation_class.value,
                workspace_revision=memory.workspace_revision,
            )

    @staticmethod
    def _decision(
        request: ApprovalRequest,
        *,
        action: ApprovalAction,
        source: str,
        reason: str,
        scope: ApprovalScope = ApprovalScope.ONCE,
        constraints: list[str] | None = None,
        guidance: str | None = None,
    ) -> ApprovalDecision:
        return ApprovalDecision(
            action=action,
            source=source,
            reason=reason,
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
            scope=scope,
            constraints=constraints or [],
            guidance=guidance,
        )

    def _task_summary(self) -> str:
        memory = self.grant_store.memory
        if memory.working.current_focus:
            return memory.working.current_focus
        if memory.task_updates:
            return memory.task_updates[-1]
        if memory.fixed is not None:
            return memory.fixed.original_task
        return ""

    def _log_request(self, request: ApprovalRequest) -> None:
        self._log(
            "approval_requested",
            request_id=request.request_id,
            tool=request.tool_name,
            fingerprint=request.fingerprint,
            risk=request.assessment.level.name.lower(),
            capabilities=sorted(item.value for item in request.assessment.capabilities),
            reasons=request.assessment.reasons,
            **self._request_details(request),
        )

    def _log_decision(
        self,
        request: ApprovalRequest,
        decision: ApprovalDecision,
    ) -> None:
        self._log(
            "approval_decided",
            request_id=request.request_id,
            tool=request.tool_name,
            fingerprint=request.fingerprint,
            action=decision.action.value,
            source=decision.source,
            reason=decision.reason,
            risk=decision.risk_level.name.lower(),
            constraints=decision.constraints,
            scope=decision.scope.value,
            abort_agent=decision.abort_agent,
            guidance=decision.guidance,
            **self._request_details(request),
        )

    def _log_review(
        self,
        request: ApprovalRequest,
        review: LLMReviewResult | None,
        *,
        duration_seconds: float,
        failure: dict[str, Any] | None = None,
    ) -> None:
        if review is None:
            self._log(
                "llm_approval_review_failed",
                request_id=request.request_id,
                tool=request.tool_name,
                fingerprint=request.fingerprint,
                duration_seconds=duration_seconds,
                **(failure or {}),
            )
            return
        self._log(
            "llm_approval_reviewed",
            request_id=request.request_id,
            tool=request.tool_name,
            fingerprint=request.fingerprint,
            recommendation=review.recommendation,
            risk=review.risk_level,
            task_relevance=review.task_relevance,
            recognized_effects=review.recognized_effects,
            unknowns=review.unknowns,
            required_constraints=review.required_constraints,
            reason=review.reason,
            user_prompt=review.user_prompt,
            duration_seconds=duration_seconds,
        )

    @staticmethod
    def _request_details(request: ApprovalRequest) -> dict[str, Any]:
        command = request.normalized_arguments.get("command")
        program = command.get("program") if isinstance(command, dict) else None
        command_args = command.get("args") if isinstance(command, dict) else None
        return {
            "affected_paths": request.assessment.affected_paths,
            "program": program if isinstance(program, str) else None,
            "argument_count": len(command_args) if isinstance(command_args, list) else None,
            "operation_class": request.facts.operation_class.value,
            "analysis_level": request.facts.analysis_level.value,
            "executable_origin": request.facts.executable_origin.value,
            "effect_scope": request.facts.effect_scope.value,
            "read_paths": request.facts.read_paths,
            "write_paths": request.facts.write_paths,
            "output_provenance": {
                path: provenance.value
                for path, provenance in request.facts.output_provenance.items()
            },
            "semantic_reasons": request.facts.reasons,
        }

    def _log(self, event_type: str, **data: Any) -> None:
        if self.event_logger is not None:
            self.event_logger.log(event_type, **data)
