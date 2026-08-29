from repo_rivet.agent.state import SessionState
from repo_rivet.llm.base import ModelResponse
from repo_rivet.tools.base import ToolCall, ToolResult


def test_record_model_response_tracks_empty_responses() -> None:
    state = SessionState(task="task")

    state.record_model_response(ModelResponse())
    state.record_model_response(ModelResponse(content="working"))

    assert state.step_count == 2
    assert state.empty_model_responses == 0
    assert len(state.messages) == 2
    assert state.messages[0]["role"] == "system"
    assert state.messages[1]["role"] == "assistant"


def test_record_tool_result_tracks_changes_failures_and_repetition() -> None:
    state = SessionState(task="task")
    call = ToolCall(
        id="call-1",
        name="edit_file",
        arguments={
            "path": "app.py",
            "snapshot_id": "a" * 64,
            "operations": [
                {
                    "op": "replace",
                    "start_line": 1,
                    "end_line": 1,
                    "new_lines": ["b"],
                }
            ],
        },
    )

    state.record_tool_result(call, ToolResult(ok=False, output="", error="not found"))
    state.record_tool_result(call, ToolResult(ok=True, output="changed"))

    assert state.tool_call_count == 2
    assert state.repeated_tool_calls == 2
    assert state.consecutive_failures == 0
    assert state.modified_files == {"app.py"}
    assert state.workspace_revision == 1
    assert state.progress_revision == 1
    assert "not found" in state.state_summary()


def test_only_new_successful_tool_observations_count_as_progress() -> None:
    state = SessionState(task="task")
    read = ToolCall(id="read-1", name="read_file", arguments={"path": "app.py"})
    result = ToolResult(ok=True, output="value = 1")

    state.record_tool_result(read, result)
    state.record_tool_result(read, result)
    state.record_tool_result(
        ToolCall(id="decision", name="record_decision", arguments={"summary": "read it"}),
        ToolResult(ok=True, output="recorded"),
    )

    assert state.progress_revision == 1
    assert state.made_progress_since_checkpoint

    state.step_count = 30
    state.renew_step_checkpoint(30)
    assert state.step_limit == 60
    assert not state.made_progress_since_checkpoint
