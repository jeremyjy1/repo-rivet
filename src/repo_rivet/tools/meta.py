"""Provider-visible meta tools that record state without local side effects."""

from repo_rivet.reasoning.models import RecordDecisionArgs
from repo_rivet.tools.base import BaseTool, ToolResult


class RecordDecisionTool(BaseTool[RecordDecisionArgs]):
    name = "record_decision"
    description = (
        "Record a concise, structured, auditable plan, decision, reflection, or final "
        "assessment. Do not include hidden chain-of-thought, secrets, source files, or raw logs."
    )
    arguments_type = RecordDecisionArgs

    def run(self, arguments: RecordDecisionArgs) -> ToolResult:
        return ToolResult(ok=True, output=f"Recorded {arguments.phase.value} summary.")
