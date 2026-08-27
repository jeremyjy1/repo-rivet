"""Non-overridable local safety decisions."""

from dataclasses import dataclass

from repo_rivet.approval.models import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalRequest,
    Capability,
    RiskLevel,
)


@dataclass(frozen=True, slots=True)
class HardSafetySettings:
    deny_outside_workspace_write: bool = True
    deny_privilege_escalation: bool = True
    deny_secret_access: bool = True
    deny_device_access: bool = True


class HardSafetyPolicy:
    def __init__(self, settings: HardSafetySettings | None = None) -> None:
        self.settings = settings or HardSafetySettings()

    def evaluate(self, request: ApprovalRequest) -> ApprovalDecision | None:
        capabilities = request.assessment.capabilities
        reason: str | None = None
        if (
            self.settings.deny_privilege_escalation
            and Capability.PRIVILEGE_ESCALATION in capabilities
        ):
            reason = "privilege escalation is prohibited"
        elif self.settings.deny_device_access and Capability.DEVICE_ACCESS in capabilities:
            reason = "device access is prohibited"
        elif self.settings.deny_secret_access and Capability.SECRET_READ in capabilities:
            reason = "credential and sensitive configuration reads are prohibited"
        elif (
            request.assessment.sensitive_paths
            and Capability.FILESYSTEM_WRITE in capabilities
        ):
            reason = "modification of sensitive or agent configuration files is prohibited"
        elif (
            self.settings.deny_outside_workspace_write
            and Capability.OUTSIDE_WORKSPACE in capabilities
            and capabilities
            & {Capability.FILESYSTEM_WRITE, Capability.FILESYSTEM_DELETE}
        ):
            reason = "writes and deletions outside the workspace are prohibited"
        elif request.assessment.level == RiskLevel.CRITICAL:
            reason = "the request has unacceptable critical risk"
        if reason is None:
            return None
        request.assessment.hard_denied = True
        return ApprovalDecision(
            action=ApprovalAction.DENY,
            source="hard_policy",
            reason=reason,
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
        )
