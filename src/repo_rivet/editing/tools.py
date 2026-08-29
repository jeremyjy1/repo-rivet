"""Provider-visible snapshot-anchored edit tool."""

from typing import Any

from repo_rivet.approval.models import Capability
from repo_rivet.editing.models import EditError, EditFileArguments
from repo_rivet.editing.runtime import EditingRuntime
from repo_rivet.tools.base import BaseTool, ToolResult
from repo_rivet.tools.filesystem import _is_sensitive_path


class EditFileTool(BaseTool[EditFileArguments]):
    name = "edit_file"
    description = (
        "Atomically edit one existing file using structured line operations anchored to a "
        "snapshot_id returned by read_file. Every target line must have been shown. All line "
        "numbers refer to the original snapshot. Supports replace, delete, insert_before, "
        "insert_after, insert_start, and insert_end; structural block editing is not supported. "
        "Keep each request to a small coherent section. Never replace an entire large file in "
        "one call; apply multiple snapshot-bound edits and reread between calls."
    )
    arguments_type = EditFileArguments
    capabilities = frozenset({Capability.FILESYSTEM_READ, Capability.FILESYSTEM_WRITE})

    def __init__(self, runtime: EditingRuntime) -> None:
        self.runtime = runtime

    def approval_arguments(self, arguments: EditFileArguments) -> dict[str, Any] | ToolResult:
        if _is_sensitive_path(arguments.path):
            return ToolResult(
                ok=False,
                output="",
                error=f"Sensitive configuration files cannot be modified: {arguments.path}",
                error_code="sensitive_path",
                retryable=False,
            )
        try:
            return self.runtime.approval_arguments(arguments)
        except EditError as error:
            return _edit_error_result(error)

    def run(self, arguments: EditFileArguments) -> ToolResult:
        if _is_sensitive_path(arguments.path):
            return ToolResult(
                ok=False,
                output="",
                error=f"Sensitive configuration files cannot be modified: {arguments.path}",
                error_code="sensitive_path",
                retryable=False,
            )
        try:
            result = self.runtime.commit(arguments)
        except EditError as error:
            return _edit_error_result(error)
        metadata = result.model_dump(mode="json")
        return ToolResult(
            ok=True,
            output=(
                f"Edited {result.path} from {result.old_snapshot_tag} to "
                f"{result.new_snapshot_tag}.\n{result.diff_preview}"
            ).rstrip(),
            metadata=metadata,
            raw_output=result.diff_preview,
        )

    def approval_granted(self, arguments: EditFileArguments, *, source: str) -> None:
        self.runtime.record_approval(arguments, source=source)


def _edit_error_result(error: EditError) -> ToolResult:
    return ToolResult(
        ok=False,
        output="",
        error=str(error),
        error_code=error.code,
        retryable=error.retryable,
        metadata=error.metadata,
    )
