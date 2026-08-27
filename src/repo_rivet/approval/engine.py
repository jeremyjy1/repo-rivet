"""Layer deterministic safety, modes, grants, LLM advice, and human decisions."""

from dataclasses import dataclass
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
    RiskLevel,
)
from repo_rivet.approval.normalizer import RequestNormalizer
from repo_rivet.approval.risk_analyzer import RiskAnalyzer


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
        minimum_llm_confidence: float = 0.90,
        max_llm_risk: RiskLevel = RiskLevel.MEDIUM,
        event_logger: EventSink | None = None,
    ) -> None:
        self.mode = mode
        self.normalizer = normalizer
        self.risk_analyzer = risk_analyzer
        self.hard_policy = hard_policy
        self.grant_store = grant_store
        self.human_approver = human_approver
        self.llm_reviewer = llm_reviewer
        self.minimum_llm_confidence = minimum_llm_confidence
        self.max_llm_risk = max_llm_risk
        self.event_logger = event_logger
        self.sync_memory_rule()

    @property
    def session_id(self) -> str:
        return self.grant_store.memory.session_id

    def set_mode(self, mode: ApprovalMode) -> None:
        """Switch policy immediately and persist the explicit session override."""
        previous_mode = self.mode
        self.mode = mode
        self.grant_store.memory.approval_mode_override = mode
        self.grant_store.memory.denied_request_fingerprints.clear()
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
        if self.mode == ApprovalMode.READ_ONLY:
            fixed.safety_rules.append(
                "Current approval mode: read-only. Only typed, workspace-confined file "
                "inspection tools are permitted; do not request writes or commands."
            )
        else:
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
        self.risk_analyzer.assess(request)
        self._log_request(request)

        decision = self.hard_policy.evaluate(request)
        if decision is None and self.mode == ApprovalMode.READ_ONLY:
            decision = self._decide_read_only(request)
        if decision is None and self.mode != ApprovalMode.ALWAYS_ASK:
            grant = self.grant_store.match(request)
            if grant is not None:
                action = ApprovalAction(grant.action)
                decision = self._decision(
                    request,
                    action=action,
                    source="session_grant",
                    reason=f"matched an exact session {grant.action} decision",
                    scope=ApprovalScope.SESSION_EXACT,
                )
            elif request.fingerprint in self.grant_store.memory.denied_request_fingerprints:
                decision = self._decision(
                    request,
                    action=ApprovalAction.DENY,
                    source="prior_denial",
                    reason="the exact request was already denied during this task",
                )
        if decision is None:
            decision = self._decide_by_mode(request)

        if decision.scope == ApprovalScope.SESSION_EXACT and self.mode != ApprovalMode.ALWAYS_ASK:
            self.grant_store.remember(request, decision.action)
        elif decision.action == ApprovalAction.DENY:
            self.grant_store.memory.denied_request_fingerprints.add(request.fingerprint)
        self._log_decision(request, decision)
        return ApprovalOutcome(request=request, decision=decision)

    def revalidate(self, outcome: ApprovalOutcome) -> ApprovalDecision | None:
        refreshed = self.normalizer.refresh(outcome.request)
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
        hard_decision = self.hard_policy.evaluate(refreshed)
        if hard_decision is not None:
            self._log_decision(refreshed, hard_decision)
            return hard_decision
        if self.mode == ApprovalMode.READ_ONLY and not refreshed.assessment.obviously_safe:
            read_only_decision = self._decide_read_only(refreshed)
            self._log_decision(refreshed, read_only_decision)
            return read_only_decision
        return None

    def record_execution(self, outcome: ApprovalOutcome, *, ok: bool, metadata: Any) -> None:
        self._log(
            "approved_tool_executed",
            request_id=outcome.request.request_id,
            tool=outcome.request.tool_name,
            fingerprint=outcome.request.fingerprint,
            ok=ok,
            metadata=metadata,
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
        if self.mode == ApprovalMode.SAFE_AUTO:
            return self.human_approver.ask(request)
        return self._review_with_llm_or_human(request)

    def _decide_read_only(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.assessment.obviously_safe:
            return self._decision(
                request,
                action=ApprovalAction.ALLOW,
                source="read_only_mode",
                reason="allowed as a typed, workspace-confined read-only operation",
            )
        return self._decision(
            request,
            action=ApprovalAction.DENY,
            source="read_only_mode",
            reason="read-only mode prohibits file changes and command execution",
        )

    def _review_with_llm_or_human(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.assessment.level > self.max_llm_risk or self.llm_reviewer is None:
            return self.human_approver.ask(request)
        try:
            review = self.llm_reviewer.review(request)
        except Exception:
            review = None
        if review is not None and self._accept_llm_approval(request, review):
            return self._decision(
                request,
                action=ApprovalAction.ALLOW,
                source="llm_reviewer",
                reason=review.reason,
                llm_confidence=review.confidence,
            )
        return self.human_approver.ask(request, llm_review=review)

    def _accept_llm_approval(
        self,
        request: ApprovalRequest,
        review: LLMReviewResult,
    ) -> bool:
        forbidden = {
            Capability.GIT_HISTORY_REWRITE,
            Capability.OUTSIDE_WORKSPACE,
            Capability.PRIVILEGE_ESCALATION,
            Capability.SECRET_READ,
        }
        return (
            review.decision == "allow"
            and review.confidence >= self.minimum_llm_confidence
            and request.assessment.level <= self.max_llm_risk
            and review.risk_level <= self.max_llm_risk
            and not request.assessment.capabilities & forbidden
        )

    @staticmethod
    def _decision(
        request: ApprovalRequest,
        *,
        action: ApprovalAction,
        source: str,
        reason: str,
        scope: ApprovalScope = ApprovalScope.ONCE,
        llm_confidence: float | None = None,
    ) -> ApprovalDecision:
        return ApprovalDecision(
            action=action,
            source=source,
            reason=reason,
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
            scope=scope,
            llm_confidence=llm_confidence,
        )

    def _log_request(self, request: ApprovalRequest) -> None:
        command = request.normalized_arguments.get("command")
        program = command.get("program") if isinstance(command, dict) else None
        command_args = command.get("args") if isinstance(command, dict) else None
        self._log(
            "approval_requested",
            request_id=request.request_id,
            tool=request.tool_name,
            fingerprint=request.fingerprint,
            risk=request.assessment.level.name.lower(),
            capabilities=sorted(item.value for item in request.assessment.capabilities),
            reasons=request.assessment.reasons,
            affected_paths=request.assessment.affected_paths,
            program=program if isinstance(program, str) else None,
            argument_count=len(command_args) if isinstance(command_args, list) else None,
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
            confidence=decision.llm_confidence,
            scope=decision.scope.value,
            abort_agent=decision.abort_agent,
        )

    def _log(self, event_type: str, **data: Any) -> None:
        if self.event_logger is not None:
            self.event_logger.log(event_type, **data)
