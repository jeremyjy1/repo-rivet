"""Read-only Git inspection tools."""

from repo_rivet.approval.models import Capability
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import BaseTool, ToolArguments, ToolResult
from repo_rivet.tools.shell import run_process


class GitDiffArguments(ToolArguments):
    path: str = "."


class GitDiffTool(BaseTool[GitDiffArguments]):
    name = "git_diff"
    description = "Show unstaged and staged Git changes for a workspace path."
    arguments_type = GitDiffArguments
    capabilities = frozenset({Capability.FILESYSTEM_READ})

    def __init__(self, path_policy: WorkspacePathPolicy) -> None:
        self.path_policy = path_policy

    def run(self, arguments: GitDiffArguments) -> ToolResult:
        resolved_path = self.path_policy.resolve(arguments.path)
        relative_path = resolved_path.relative_to(self.path_policy.workspace)
        pathspec = relative_path.as_posix() or "."

        unstaged = run_process(
            ("git", "diff", "--no-ext-diff", "--", pathspec),
            cwd=self.path_policy.workspace,
            timeout_seconds=30,
        )
        if not unstaged.ok:
            return unstaged
        if unstaged.metadata and unstaged.metadata.get("exit_code") != 0:
            return ToolResult(
                ok=False,
                output=unstaged.output,
                error=f"git diff failed with exit code {unstaged.metadata['exit_code']}",
                metadata=unstaged.metadata,
            )
        staged = run_process(
            ("git", "diff", "--cached", "--no-ext-diff", "--", pathspec),
            cwd=self.path_policy.workspace,
            timeout_seconds=30,
        )
        if not staged.ok:
            return staged
        if staged.metadata and staged.metadata.get("exit_code") != 0:
            return ToolResult(
                ok=False,
                output=staged.output,
                error=f"git diff --cached failed with exit code {staged.metadata['exit_code']}",
                metadata=staged.metadata,
            )

        output = f"UNSTAGED:\n{unstaged.output}\n\nSTAGED:\n{staged.output}"
        return ToolResult(
            ok=True,
            output=output,
            metadata={
                "path": pathspec,
                "unstaged": unstaged.metadata,
                "staged": staged.metadata,
            },
            raw_output=f"UNSTAGED:\n{unstaged.raw_output or unstaged.output}\n\n"
            f"STAGED:\n{staged.raw_output or staged.output}",
        )
