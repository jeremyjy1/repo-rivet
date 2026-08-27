from pathlib import Path

import pytest

from repo_rivet.safety.path_policy import PathPolicyError, WorkspacePathPolicy


def test_resolve_path_inside_workspace(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    policy = WorkspacePathPolicy(tmp_path)

    assert policy.resolve("src/module.py") == source / "module.py"
    assert policy.relative("src/module.py") == Path("src/module.py")


def test_reject_absolute_path(tmp_path: Path) -> None:
    policy = WorkspacePathPolicy(tmp_path)

    with pytest.raises(PathPolicyError, match="Absolute paths"):
        policy.resolve(tmp_path / "file.py")


def test_reject_parent_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspacePathPolicy(workspace)

    with pytest.raises(PathPolicyError, match="escapes workspace"):
        policy.resolve("../outside.txt")


def test_reject_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    policy = WorkspacePathPolicy(workspace)

    with pytest.raises(PathPolicyError, match="escapes workspace"):
        policy.resolve("link/secret.txt")


def test_reject_non_directory_workspace(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(PathPolicyError, match="not a directory"):
        WorkspacePathPolicy(file_path)
