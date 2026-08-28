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
    assert "not found" in state.state_summary()
