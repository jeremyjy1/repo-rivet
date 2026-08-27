"""Exact-request approval and denial grants scoped to one persisted session."""

from datetime import UTC, datetime

from repo_rivet.approval.models import ApprovalAction, ApprovalGrant, ApprovalRequest
from repo_rivet.memory.models import MemoryState


class ApprovalGrantStore:
    def __init__(
        self,
        memory: MemoryState,
        *,
        remember_approvals: bool = True,
        remember_denials: bool = True,
    ) -> None:
        self.memory = memory
        self.remember_approvals = remember_approvals
        self.remember_denials = remember_denials

    def match(self, request: ApprovalRequest) -> ApprovalGrant | None:
        payload = self.memory.approval_session_grants.get(request.fingerprint)
        if payload is None:
            return None
        grant = ApprovalGrant.model_validate(payload)
        if grant.session_id != request.session_id:
            return None
        if grant.expires_at is not None and grant.expires_at <= datetime.now(UTC):
            self.memory.approval_session_grants.pop(request.fingerprint, None)
            return None
        return grant

    def remember(self, request: ApprovalRequest, action: ApprovalAction) -> None:
        if action == ApprovalAction.ALLOW and not self.remember_approvals:
            return
        if action == ApprovalAction.DENY and not self.remember_denials:
            return
        if action not in {ApprovalAction.ALLOW, ApprovalAction.DENY}:
            return
        grant = ApprovalGrant(
            request_fingerprint=request.fingerprint,
            session_id=request.session_id,
            action=action.value,
        )
        self.memory.approval_session_grants[request.fingerprint] = grant.model_dump(mode="json")
        if action == ApprovalAction.DENY:
            self.memory.denied_request_fingerprints.add(request.fingerprint)
