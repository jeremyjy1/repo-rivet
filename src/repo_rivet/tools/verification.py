"""Provider-visible tools for explicit verification plans and registered checks."""

from pydantic import Field

from repo_rivet.approval.models import Capability
from repo_rivet.tools.base import BaseTool, ToolArguments, ToolResult
from repo_rivet.verification.models import VerificationCheck
from repo_rivet.verification.runtime import VerificationRuntime


class RegisterVerificationArguments(ToolArguments):
    requirements: list[str] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "Optional acceptance-criterion identifiers. Each value must equal a required "
            "check_id or appear in that check's claim_ids."
        ),
    )
    checks: list[VerificationCheck] = Field(min_length=1, max_length=100)


class RegisterVerificationTool(BaseTool[RegisterVerificationArguments]):
    name = "register_verification"
    description = (
        "Register explicit required verification checks before changing files. Checks use "
        "shell-free program/args commands and deterministic success criteria. Requirements "
        "may directly reference required check IDs; claim_ids are only needed when a "
        "requirement uses a different identifier."
    )
    arguments_type = RegisterVerificationArguments

    def run(self, arguments: RegisterVerificationArguments) -> ToolResult:
        return ToolResult(ok=True, output="Verification plan schema is valid.")


class RunVerificationArguments(ToolArguments):
    check_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


class RunVerificationTool(BaseTool[RunVerificationArguments]):
    name = "run_verification"
    description = (
        "Run one previously registered verification check by ID. The local controller "
        "evaluates its declared success criteria; arbitrary commands cannot claim verification."
    )
    arguments_type = RunVerificationArguments
    capabilities = frozenset({Capability.PROCESS_EXECUTE})

    def __init__(self, runtime: VerificationRuntime) -> None:
        self.runtime = runtime

    def validate_arguments(
        self,
        arguments: dict[str, object],
    ) -> RunVerificationArguments | ToolResult:
        validated = super().validate_arguments(arguments)
        if isinstance(validated, ToolResult):
            return validated
        try:
            self.runtime.check(validated.check_id)
        except ValueError as error:
            return ToolResult(
                ok=False,
                output="",
                error=str(error),
                error_code="verification_check_invalid",
                retryable=True,
            )
        return validated

    def approval_arguments(self, arguments: RunVerificationArguments) -> dict[str, object]:
        return self.runtime.approval_arguments(arguments.check_id)

    def run(self, arguments: RunVerificationArguments) -> ToolResult:
        return self.runtime.run(arguments.check_id)
