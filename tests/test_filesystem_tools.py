from pathlib import Path

from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.filesystem import (
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
    SearchTextTool,
    WriteFileTool,
)


def test_list_search_and_read_files(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "service.py").write_text("first\nneedle here\nlast\n", encoding="utf-8")
    policy = WorkspacePathPolicy(tmp_path)

    listed = ListFilesTool(policy).execute({"path": ".", "max_depth": 2})
    searched = SearchTextTool(policy).execute({"query": "needle", "path": "src"})
    read = ReadFileTool(policy).execute({"path": "src/service.py", "start_line": 2, "end_line": 3})

    assert listed.ok and "src/service.py" in listed.output
    assert searched.ok and "src/service.py:2:needle here" in searched.output
    assert read.ok and "2 | needle here" in read.output and "3 | last" in read.output


def test_write_file_requires_explicit_overwrite(tmp_path: Path) -> None:
    tool = WriteFileTool(WorkspacePathPolicy(tmp_path))

    created = tool.execute({"path": "src/new.py", "content": "old"})
    rejected = tool.execute({"path": "src/new.py", "content": "new"})
    overwritten = tool.execute({"path": "src/new.py", "content": "new", "overwrite": True})

    assert created.ok
    assert not rejected.ok and "overwrite=true" in (rejected.error or "")
    assert overwritten.ok
    assert (tmp_path / "src/new.py").read_text(encoding="utf-8") == "new"


def test_replace_text_checks_expected_count_before_writing(tmp_path: Path) -> None:
    file_path = tmp_path / "module.py"
    file_path.write_text("old old", encoding="utf-8")
    tool = ReplaceTextTool(WorkspacePathPolicy(tmp_path))

    rejected = tool.execute(
        {"path": "module.py", "old_text": "old", "new_text": "new", "expected_count": 1}
    )
    replaced = tool.execute(
        {"path": "module.py", "old_text": "old", "new_text": "new", "expected_count": 2}
    )

    assert not rejected.ok
    assert file_path.read_text(encoding="utf-8") == "new new"
    assert replaced.ok


def test_file_tools_reject_workspace_escape(tmp_path: Path) -> None:
    result = ReadFileTool(WorkspacePathPolicy(tmp_path)).execute({"path": "../secret.txt"})

    assert not result.ok
    assert "escapes workspace" in (result.error or "")


def test_read_file_rejects_binary_content(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"text\x00data")

    result = ReadFileTool(WorkspacePathPolicy(tmp_path)).execute({"path": "binary.dat"})

    assert not result.ok
    assert "Binary files" in (result.error or "")


def test_read_file_rejects_sensitive_configuration(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("API_KEY=secret", encoding="utf-8")

    result = ReadFileTool(WorkspacePathPolicy(tmp_path)).execute({"path": ".env"})

    assert not result.ok
    assert "Sensitive configuration" in (result.error or "")
