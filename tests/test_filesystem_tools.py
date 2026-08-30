import stat
from pathlib import Path

import pytest

from repo_rivet.editing.models import EditFileArguments
from repo_rivet.editing.runtime import EditingRuntime
from repo_rivet.editing.tools import EditFileTool
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.filesystem import (
    DeletePathTool,
    ListFilesTool,
    ReadFileTool,
    SearchTextTool,
    WriteFileTool,
)


def editing_tools(
    workspace: Path,
    *,
    snapshot_dir: Path | None = None,
) -> tuple[ReadFileTool, EditFileTool, EditingRuntime]:
    policy = WorkspacePathPolicy(workspace)
    runtime = EditingRuntime(policy, snapshot_dir=snapshot_dir)
    return ReadFileTool(policy, runtime), EditFileTool(runtime), runtime


def read_snapshot(read: ReadFileTool, path: str, start: int = 1, end: int | None = None) -> str:
    result = read.execute({"path": path, "start_line": start, "end_line": end})
    assert result.ok and result.metadata
    return str(result.metadata["snapshot_id"])


def test_list_search_and_read_files_return_snapshot_anchors(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "service.py").write_text("first\nneedle here\nlast\n", encoding="utf-8")
    policy = WorkspacePathPolicy(tmp_path)
    runtime = EditingRuntime(policy)

    listed = ListFilesTool(policy).execute({"path": ".", "max_depth": 2})
    searched = SearchTextTool(policy, runtime).execute({"query": "needle", "path": "src"})
    read = ReadFileTool(policy, runtime).execute(
        {"path": "src/service.py", "start_line": 2, "end_line": 3}
    )

    assert listed.ok and "src/service.py" in listed.output
    assert searched.ok and "src/service.py:2:needle here" in searched.output
    assert searched.metadata and searched.metadata["snapshot_ids"]["src/service.py"]
    assert read.ok and "2│ needle here" in read.output and "3│ last" in read.output
    assert read.metadata and read.metadata["snapshot_tag"] in read.output.splitlines()[0]


def test_unchanged_reads_reuse_snapshot_and_external_change_creates_new_one(
    tmp_path: Path,
) -> None:
    path = tmp_path / "module.py"
    path.write_text("one\ntwo\n", encoding="utf-8")
    read, _, _ = editing_tools(tmp_path)

    first = read_snapshot(read, "module.py", 1, 2)
    second = read_snapshot(read, "module.py", 1, 2)
    path.write_text("one\nchanged\n", encoding="utf-8")
    changed = read_snapshot(read, "module.py", 1, 2)

    assert first == second
    assert changed != first


def test_write_file_only_creates_new_paths(tmp_path: Path) -> None:
    tool = WriteFileTool(WorkspacePathPolicy(tmp_path))

    created = tool.execute({"path": "src/new.py", "content": "old"})
    rejected = tool.execute({"path": "src/new.py", "content": "new"})

    assert created.ok and created.metadata and created.metadata["snapshot_id"]
    assert not rejected.ok and "edit_file" in (rejected.error or "")
    assert (tmp_path / "src/new.py").read_text(encoding="utf-8") == "old"


def test_delete_path_removes_workspace_file_and_empty_directory(tmp_path: Path) -> None:
    policy = WorkspacePathPolicy(tmp_path)
    runtime = EditingRuntime(policy)
    tool = DeletePathTool(policy, runtime)
    file_path = tmp_path / "obsolete.txt"
    empty_directory = tmp_path / "empty"
    file_path.write_text("obsolete\n", encoding="utf-8")
    empty_directory.mkdir()

    deleted_file = tool.execute({"path": "obsolete.txt"})
    deleted_directory = tool.execute({"path": "empty"})

    assert deleted_file.ok and deleted_file.metadata
    assert deleted_file.metadata["path_type"] == "file"
    assert deleted_file.metadata["workspace_revision"] == 1
    assert deleted_directory.ok and deleted_directory.metadata
    assert deleted_directory.metadata["path_type"] == "directory"
    assert deleted_directory.metadata["workspace_revision"] == 2
    assert not file_path.exists()
    assert not empty_directory.exists()


def test_delete_path_requires_explicit_recursive_directory_deletion(tmp_path: Path) -> None:
    directory = tmp_path / "generated"
    directory.mkdir()
    (directory / "one.txt").write_text("one", encoding="utf-8")
    nested = directory / "nested"
    nested.mkdir()
    (nested / "two.txt").write_text("two", encoding="utf-8")
    tool = DeletePathTool(WorkspacePathPolicy(tmp_path))

    rejected = tool.execute({"path": "generated"})
    assert not rejected.ok
    assert "recursive=true" in (rejected.error or "")
    assert directory.exists()

    deleted = tool.execute({"path": "generated", "recursive": True})

    assert deleted.ok and deleted.metadata
    assert deleted.metadata["entry_count"] == 3
    assert not directory.exists()


def test_delete_path_unlinks_symlink_without_following_target(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("keep", encoding="utf-8")
    link = tmp_path / "outside-link"
    link.symlink_to(outside)
    tool = DeletePathTool(WorkspacePathPolicy(tmp_path))

    result = tool.execute({"path": "outside-link"})

    assert result.ok and result.metadata
    assert result.metadata["path_type"] == "symlink"
    assert not link.exists()
    assert outside.read_text(encoding="utf-8") == "keep"


def test_delete_path_rejects_workspace_root_and_git_metadata(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    tool = DeletePathTool(WorkspacePathPolicy(tmp_path))

    root = tool.execute({"path": ".", "recursive": True})
    metadata = tool.execute({"path": ".git", "recursive": True})

    assert not root.ok and "workspace root" in (root.error or "")
    assert not metadata.ok and "metadata" in (metadata.error or "")
    assert (tmp_path / ".git").exists()


def test_recursive_delete_rejects_protected_nested_entries(tmp_path: Path) -> None:
    directory = tmp_path / "archive"
    metadata = directory / "nested" / ".git"
    metadata.mkdir(parents=True)
    tool = DeletePathTool(WorkspacePathPolicy(tmp_path))

    result = tool.execute({"path": "archive", "recursive": True})

    assert not result.ok
    assert "Protected" in (result.error or "")
    assert metadata.exists()


def test_delete_path_rejects_escape_through_parent_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    target = outside / "keep.txt"
    target.write_text("keep", encoding="utf-8")
    (tmp_path / "linked-directory").symlink_to(outside, target_is_directory=True)
    tool = DeletePathTool(WorkspacePathPolicy(tmp_path))

    result = tool.execute({"path": "linked-directory/keep.txt"})

    assert not result.ok
    assert "escapes workspace" in (result.error or "")
    assert target.exists()


def test_delete_path_revalidates_target_after_approval_preflight(tmp_path: Path) -> None:
    path = tmp_path / "changing.txt"
    path.write_text("before", encoding="utf-8")
    tool = DeletePathTool(WorkspacePathPolicy(tmp_path))
    arguments = tool.arguments_type.model_validate({"path": "changing.txt"})
    prepared = tool.approval_arguments(arguments)
    assert isinstance(prepared, dict)
    path.write_text("different content", encoding="utf-8")

    result = tool.execute_validated(arguments)

    assert not result.ok
    assert "changed during approval" in (result.error or "")
    assert path.exists()


def test_edit_operations_use_original_snapshot_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("A\nB\nC\nD\nE\n", encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "module.py", 1, 5)

    result = edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [
                {
                    "op": "replace",
                    "start_line": 1,
                    "end_line": 1,
                    "new_lines": ["X", "Y", "Z"],
                },
                {"op": "replace", "start_line": 5, "end_line": 5, "new_lines": ["Q"]},
            ],
        }
    )

    assert result.ok and result.metadata
    assert path.read_text(encoding="utf-8") == "X\nY\nZ\nB\nC\nD\nQ\n"
    assert result.metadata["old_snapshot_id"] == snapshot_id
    assert result.metadata["new_snapshot_id"] != snapshot_id


def test_insert_and_delete_operations_are_applied_atomically(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("A\nB\nC\nD\n", encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "module.py", 1, 4)

    result = edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [
                {"op": "insert_start", "new_lines": ["START"]},
                {"op": "insert_before", "line": 2, "new_lines": ["BEFORE"]},
                {"op": "delete", "start_line": 3, "end_line": 3},
                {"op": "insert_after", "line": 4, "new_lines": ["END"]},
            ],
        }
    )

    assert result.ok
    assert path.read_text(encoding="utf-8") == "START\nA\nBEFORE\nB\nD\nEND\n"


def test_file_end_insert_preserves_missing_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("A\nB", encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "module.py", 1, 2)

    result = edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [
                {"op": "replace", "start_line": 2, "end_line": 2, "new_lines": ["changed"]},
                {"op": "insert_end", "new_lines": ["C"]},
            ],
        }
    )

    assert result.ok
    assert path.read_bytes() == b"A\nchanged\nC"


def test_line_out_of_bounds_is_rejected_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("A\nB\n", encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "module.py", 1, 2)

    result = edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [{"op": "replace", "start_line": 3, "end_line": 3, "new_lines": ["C"]}],
        }
    )

    assert not result.ok and result.error_code == "line_out_of_bounds"
    assert path.read_text(encoding="utf-8") == "A\nB\n"


def test_overlapping_and_unseen_edits_are_rejected_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    original = "A\nB\nC\n"
    path.write_text(original, encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "module.py", 1, 1)

    unseen = edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [{"op": "replace", "start_line": 2, "end_line": 2, "new_lines": ["X"]}],
        }
    )
    read_snapshot(read, "module.py", 1, 3)
    overlapping = edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [
                {"op": "replace", "start_line": 1, "end_line": 2, "new_lines": ["X"]},
                {"op": "delete", "start_line": 2, "end_line": 3},
            ],
        }
    )

    assert unseen.error_code == "unseen_range"
    assert overlapping.error_code == "overlapping_operations"
    assert path.read_text(encoding="utf-8") == original


def test_stale_snapshot_and_path_mismatch_fail_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("old\n", encoding="utf-8")
    second.write_text("other\n", encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "first.py", 1, 1)

    mismatch = edit.execute(
        {
            "path": "second.py",
            "snapshot_id": snapshot_id,
            "operations": [{"op": "replace", "start_line": 1, "end_line": 1, "new_lines": ["new"]}],
        }
    )
    first.write_text("external\n", encoding="utf-8")
    stale = edit.execute(
        {
            "path": "first.py",
            "snapshot_id": snapshot_id,
            "operations": [{"op": "replace", "start_line": 1, "end_line": 1, "new_lines": ["new"]}],
        }
    )

    assert mismatch.error_code == "snapshot_path_mismatch"
    assert stale.error_code == "stale_snapshot"
    assert stale.metadata and stale.metadata["current_total_lines"] == 1
    assert first.read_text(encoding="utf-8") == "external\n"
    assert second.read_text(encoding="utf-8") == "other\n"


def test_search_result_makes_only_matching_line_editable(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("first\nneedle\nlast\n", encoding="utf-8")
    policy = WorkspacePathPolicy(tmp_path)
    runtime = EditingRuntime(policy)
    search = SearchTextTool(policy, runtime)
    edit = EditFileTool(runtime)

    result = search.execute({"path": "module.py", "query": "needle"})
    assert result.metadata
    snapshot_id = result.metadata["snapshot_ids"]["module.py"]
    allowed = edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [
                {"op": "replace", "start_line": 2, "end_line": 2, "new_lines": ["found"]}
            ],
        }
    )

    assert allowed.ok
    assert path.read_text(encoding="utf-8") == "first\nfound\nlast\n"


def test_repeated_noop_triggers_loop_guard(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("same\n", encoding="utf-8")
    snapshot_dir = tmp_path / "session" / "snapshots"
    read, edit, _ = editing_tools(tmp_path, snapshot_dir=snapshot_dir)
    snapshot_id = read_snapshot(read, "module.py", 1, 1)
    request = {
        "path": "module.py",
        "snapshot_id": snapshot_id,
        "operations": [{"op": "replace", "start_line": 1, "end_line": 1, "new_lines": ["same"]}],
    }

    first = edit.execute(request)
    second = edit.execute(request)
    _, resumed_edit, _ = editing_tools(tmp_path, snapshot_dir=snapshot_dir)
    third = resumed_edit.execute(request)

    assert first.error_code == "edit_noop"
    assert second.error_code == "edit_noop" and "reread" in (second.error or "")
    assert third.error_code == "edit_loop_detected" and third.retryable is False


def test_unicode_line_separator_is_not_treated_as_a_file_line_break(tmp_path: Path) -> None:
    path = tmp_path / "module.txt"
    path.write_text("left\u2028right\nlast\n", encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "module.txt", 1, 2)

    result = edit.execute(
        {
            "path": "module.txt",
            "snapshot_id": snapshot_id,
            "operations": [
                {"op": "replace", "start_line": 2, "end_line": 2, "new_lines": ["changed"]}
            ],
        }
    )

    assert result.ok
    assert path.read_text(encoding="utf-8") == "left\u2028right\nchanged\n"


def test_partially_displayed_long_line_is_not_editable(tmp_path: Path) -> None:
    path = tmp_path / "module.txt"
    path.write_text("x" * 25_000 + "\n", encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    read_result = read.execute({"path": "module.txt", "start_line": 1, "end_line": 1})
    assert read_result.ok and read_result.metadata

    result = edit.execute(
        {
            "path": "module.txt",
            "snapshot_id": read_result.metadata["snapshot_id"],
            "operations": [
                {"op": "replace", "start_line": 1, "end_line": 1, "new_lines": ["changed"]}
            ],
        }
    )

    assert read_result.metadata["fully_visible_end_line"] is None
    assert not result.ok and result.error_code == "unseen_range"


def test_empty_file_snapshot_allows_file_start_insertion(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "empty.txt")

    result = edit.execute(
        {
            "path": "empty.txt",
            "snapshot_id": snapshot_id,
            "operations": [{"op": "insert_start", "new_lines": ["first"]}],
        }
    )

    assert result.ok
    assert path.read_bytes() == b"first"


def test_edit_preserves_bom_crlf_trailing_newline_and_permissions(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_bytes(b"\xef\xbb\xbfA\r\nB\r\n")
    path.chmod(0o640)
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "module.py", 1, 2)

    result = edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [{"op": "replace", "start_line": 2, "end_line": 2, "new_lines": ["C"]}],
        }
    )

    assert result.ok
    assert path.read_bytes() == b"\xef\xbb\xbfA\r\nC\r\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_change_during_approval_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("old\n", encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "module.py", 1, 1)
    arguments = EditFileArguments.model_validate(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [{"op": "replace", "start_line": 1, "end_line": 1, "new_lines": ["new"]}],
        }
    )
    prepared = edit.approval_arguments(arguments)
    path.write_text("human\n", encoding="utf-8")

    result = edit.run(arguments)

    assert isinstance(prepared, dict) and "diff_preview" in prepared
    assert not result.ok and result.error_code == "edit_changed_during_approval"
    assert path.read_text(encoding="utf-8") == "human\n"


def test_atomic_replace_failure_keeps_original_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "module.py"
    path.write_text("old\n", encoding="utf-8")
    read, edit, _ = editing_tools(tmp_path)
    snapshot_id = read_snapshot(read, "module.py", 1, 1)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("repo_rivet.editing.atomic_writer.os.replace", fail_replace)
    result = edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [{"op": "replace", "start_line": 1, "end_line": 1, "new_lines": ["new"]}],
        }
    )

    assert not result.ok and result.error_code == "io_error"
    assert path.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".*.reporivet.tmp")) == []


def test_snapshots_and_visibility_survive_runtime_restart(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    snapshots = tmp_path / "session" / "snapshots"
    workspace.mkdir()
    path = workspace / "module.py"
    path.write_text("old\n", encoding="utf-8")
    read, _, _ = editing_tools(workspace, snapshot_dir=snapshots)
    snapshot_id = read_snapshot(read, "module.py", 1, 1)
    _, resumed_edit, _ = editing_tools(workspace, snapshot_dir=snapshots)

    result = resumed_edit.execute(
        {
            "path": "module.py",
            "snapshot_id": snapshot_id,
            "operations": [{"op": "replace", "start_line": 1, "end_line": 1, "new_lines": ["new"]}],
        }
    )

    assert result.ok
    assert path.read_text(encoding="utf-8") == "new\n"


def test_file_tools_reject_workspace_escape(tmp_path: Path) -> None:
    result = ReadFileTool(WorkspacePathPolicy(tmp_path)).execute({"path": "../secret.txt"})

    assert not result.ok
    assert "escapes workspace" in (result.error or "")


def test_read_file_rejects_binary_and_sensitive_content(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"text\x00data")
    (tmp_path / ".env").write_text("API_KEY=secret", encoding="utf-8")
    tool = ReadFileTool(WorkspacePathPolicy(tmp_path))

    binary = tool.execute({"path": "binary.dat"})
    sensitive = tool.execute({"path": ".env"})

    assert binary.error_code == "unsupported_text_encoding"
    assert "Sensitive configuration" in (sensitive.error or "")
