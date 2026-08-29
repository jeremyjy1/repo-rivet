from pathlib import Path

from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.tools.base import DecisionPolicy, ToolCall
from repo_rivet.tools.git import GitDiffTool, GitStatusTool
from repo_rivet.tools.registry import create_default_registry


def test_default_registry_exposes_workspace_and_decision_tools(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    assert registry.names == (
        "record_decision",
        "register_verification",
        "submit_plan",
        "update_plan",
        "request_plan",
        "list_files",
        "search_text",
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
        "run_verification",
        "git_status",
        "git_diff",
    )
    assert [schema["function"]["name"] for schema in registry.schemas()] == list(registry.names)


def test_registry_returns_structured_error_for_unknown_tool(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = registry.execute(ToolCall(id="call-1", name="missing", arguments={}))

    assert not result.ok
    assert "Unknown tool" in (result.error or "")
    assert "read_file" in (result.error or "")


def test_registry_validates_tool_arguments(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = registry.execute(
        ToolCall(id="call-1", name="read_file", arguments={"path": "file", "unknown": True})
    )

    assert not result.ok
    assert "Invalid arguments" in (result.error or "")


def test_registry_identifies_workspace_file_mutation_semantically(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    assert registry.modifies_workspace_files("write_file")
    assert registry.modifies_workspace_files("edit_file")
    assert not registry.modifies_workspace_files("run_command")
    assert not registry.modifies_workspace_files("read_file")


def test_registry_declares_semantic_decision_policies(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    assert registry.decision_policy("edit_file") == DecisionPolicy.MUTATION
    assert registry.decision_policy("run_command") == DecisionPolicy.COMMAND
    assert registry.decision_policy("run_verification") == DecisionPolicy.REGISTERED_PLAN


def test_run_verification_rejects_model_supplied_command(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = registry.execute(
        ToolCall(
            id="call-verify",
            name="run_verification",
            arguments={"check_id": "tests", "command": "anything"},
        )
    )

    assert not result.ok
    assert result.error_code == "invalid_arguments"


def test_git_diff_reports_non_repository_as_failure(tmp_path: Path) -> None:
    result = GitDiffTool(WorkspacePathPolicy(tmp_path)).execute({"path": "."})

    assert not result.ok
    assert "git diff failed" in (result.error or "")


def test_git_status_reports_non_repository_as_failure(tmp_path: Path) -> None:
    result = GitStatusTool(WorkspacePathPolicy(tmp_path)).execute({"path": "."})

    assert not result.ok
    assert "git status failed" in (result.error or "")
