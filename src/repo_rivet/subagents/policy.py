"""Capability and path policies for read-only child runs."""

from __future__ import annotations

from pathlib import Path

from repo_rivet.safety.path_policy import PathPolicyError, WorkspacePathPolicy
from repo_rivet.subagents.models import AgentRuntimeConfig, RunKind, SubagentProfile

_PROFILE_TOOLS: dict[SubagentProfile, frozenset[str]] = {
    SubagentProfile.EXPLORER: frozenset(
        {
            "list_files",
            "read_file",
            "search_text",
            "semantic_query",
            "git_diff",
            "submit_subagent_report",
        }
    ),
    SubagentProfile.TEST_ANALYST: frozenset(
        {
            "read_file",
            "search_text",
            "semantic_query",
            "read_tool_output",
            "submit_subagent_report",
        }
    ),
    SubagentProfile.REVIEWER: frozenset(
        {
            "read_file",
            "semantic_query",
            "git_diff",
            "read_verification_result",
            "submit_subagent_report",
        }
    ),
}


def profile_runtime_config(
    profile: SubagentProfile,
    scope_paths: list[str],
    *,
    max_model_calls: int = 4,
    max_tool_calls: int = 10,
    max_runtime_seconds: float = 90,
) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        run_kind=RunKind.SUBAGENT,
        allowed_tools=_PROFILE_TOOLS[profile],
        allowed_paths=tuple(scope_paths),
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
        max_runtime_seconds=max_runtime_seconds,
    )


class ScopedWorkspacePathPolicy(WorkspacePathPolicy):
    """Confine a child to explicit paths inside the parent workspace."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        allowed_paths: list[str],
        excluded_paths: list[str] | None = None,
    ) -> None:
        super().__init__(workspace)
        self._allowed = tuple(WorkspacePathPolicy.resolve(self, path) for path in allowed_paths)
        self._excluded = tuple(
            WorkspacePathPolicy.resolve(self, path) for path in (excluded_paths or [])
        )
        if not self._allowed:
            raise PathPolicyError("Subagent scope must include at least one path")

    def resolve(self, user_path: str | Path = ".") -> Path:
        resolved = super().resolve(user_path)
        if not any(resolved == root or resolved.is_relative_to(root) for root in self._allowed):
            raise PathPolicyError(f"Path is outside the delegated scope: {user_path}")
        if any(resolved == root or resolved.is_relative_to(root) for root in self._excluded):
            raise PathPolicyError(f"Path is excluded from the delegated scope: {user_path}")
        return resolved

    def resolve_entry(self, user_path: str | Path) -> Path:
        resolved = super().resolve_entry(user_path)
        if not any(resolved == root or resolved.is_relative_to(root) for root in self._allowed):
            raise PathPolicyError(f"Path is outside the delegated scope: {user_path}")
        if any(resolved == root or resolved.is_relative_to(root) for root in self._excluded):
            raise PathPolicyError(f"Path is excluded from the delegated scope: {user_path}")
        return resolved


def normalize_scope(workspace: Path, scope_paths: list[str]) -> list[str]:
    policy = WorkspacePathPolicy(workspace)
    resolved: list[Path] = []
    for path in scope_paths:
        value = policy.resolve(path)
        if not value.exists():
            raise PathPolicyError(f"Delegated scope does not exist: {path}")
        if value not in resolved:
            resolved.append(value)
    if not resolved:
        raise PathPolicyError("Subagent scope must include at least one path")
    minimal = [
        candidate
        for candidate in resolved
        if not any(candidate != other and candidate.is_relative_to(other) for other in resolved)
    ]
    return [candidate.relative_to(policy.workspace).as_posix() or "." for candidate in minimal]
