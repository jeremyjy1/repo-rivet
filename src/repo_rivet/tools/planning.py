"""Provider-visible tools for structured plan artifacts."""

from pydantic import Field

from repo_rivet.planning.models import PlanDraft
from repo_rivet.tools.base import BaseTool, ToolArguments, ToolResult


class SubmitPlanArguments(ToolArguments):
    plan: PlanDraft


class SubmitPlanTool(BaseTool[SubmitPlanArguments]):
    name = "submit_plan"
    description = (
        "Submit a structured, evidence-backed implementation plan for local validation and user "
        "review. This does not authorize or execute any action."
    )
    arguments_type = SubmitPlanArguments

    def run(self, arguments: SubmitPlanArguments) -> ToolResult:
        return ToolResult(ok=True, output="Plan schema is valid.")


class UpdatePlanArguments(ToolArguments):
    reason: str = Field(min_length=1, max_length=1_000)
    plan: PlanDraft


class UpdatePlanTool(BaseTool[UpdatePlanArguments]):
    name = "update_plan"
    description = (
        "Replace the current plan with a fully revised structured plan, explaining the change. "
        "The revised plan returns to user review and grants no tool approval."
    )
    arguments_type = UpdatePlanArguments

    def run(self, arguments: UpdatePlanArguments) -> ToolResult:
        return ToolResult(ok=True, output="Updated plan schema is valid.")


class RequestPlanArguments(ToolArguments):
    reason: str = Field(min_length=1, max_length=1_000)
    expected_scope: str = Field(min_length=1, max_length=500)


class RequestPlanTool(BaseTool[RequestPlanArguments]):
    name = "request_plan"
    description = (
        "Request a Controller-enforced transition into read-only Plan Mode before taking any "
        "action. Use this as the only operation in the response when the task has uncertain, "
        "multi-file, architectural, migration, or recovery scope. Planning still requires user "
        "review and grants no execution or approval rights."
    )
    arguments_type = RequestPlanArguments

    def run(self, arguments: RequestPlanArguments) -> ToolResult:
        return ToolResult(ok=True, output="Plan Mode requested.")
