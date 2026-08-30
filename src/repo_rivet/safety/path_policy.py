"""Workspace path confinement for file and command tools."""

import os
from pathlib import Path


class PathPolicyError(ValueError):
    """Raised when a requested path is outside the configured workspace."""


class WorkspacePathPolicy:
    """Resolve user paths while confining them to one workspace directory."""

    def __init__(self, workspace: str | Path) -> None:
        workspace_path = Path(workspace).expanduser()
        try:
            resolved_workspace = workspace_path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise PathPolicyError(f"Workspace does not exist: {workspace_path}") from error

        if not resolved_workspace.is_dir():
            raise PathPolicyError(f"Workspace is not a directory: {workspace_path}")
        self._workspace = resolved_workspace

    @property
    def workspace(self) -> Path:
        """Return the normalized workspace root."""
        return self._workspace

    def resolve(self, user_path: str | Path = ".") -> Path:
        """Resolve a relative path and reject absolute paths or workspace escapes."""
        requested_path = Path(user_path)
        if requested_path.is_absolute():
            raise PathPolicyError(f"Absolute paths are not allowed: {user_path}")

        try:
            resolved_path = (self._workspace / requested_path).resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise PathPolicyError(f"Could not resolve workspace path: {user_path}") from error

        if not resolved_path.is_relative_to(self._workspace):
            raise PathPolicyError(f"Path escapes workspace: {user_path}")
        return resolved_path

    def relative(self, path: str | Path) -> Path:
        """Return a normalized path relative to the workspace root."""
        return self.resolve(path).relative_to(self._workspace)

    def resolve_entry(self, user_path: str | Path) -> Path:
        """Resolve an entry without following its final symlink.

        Deletion needs to unlink a workspace symlink itself, not the file or directory it
        references. Parent symlinks are still resolved and confined to the workspace.
        """
        requested_path = Path(user_path)
        if requested_path.is_absolute():
            raise PathPolicyError(f"Absolute paths are not allowed: {user_path}")

        candidate = Path(os.path.normpath(self._workspace / requested_path))
        if not candidate.is_relative_to(self._workspace):
            raise PathPolicyError(f"Path escapes workspace: {user_path}")
        if candidate == self._workspace:
            return candidate
        try:
            resolved_parent = candidate.parent.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise PathPolicyError(f"Could not resolve workspace path: {user_path}") from error
        if not resolved_parent.is_relative_to(self._workspace):
            raise PathPolicyError(f"Path escapes workspace: {user_path}")
        return resolved_parent / candidate.name
