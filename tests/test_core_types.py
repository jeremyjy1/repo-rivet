from repo_rivet.llm.base import ModelResponse
from repo_rivet.tools.base import ToolCall, ToolResult


def test_model_response_has_independent_tool_call_lists() -> None:
    first = ModelResponse()
    second = ModelResponse()

    assert first.tool_calls == []
    assert second.tool_calls == []
    assert first.tool_calls is not second.tool_calls


def test_tool_data_structures_preserve_normalized_values() -> None:
    call = ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"})
    result = ToolResult(ok=True, output="contents", metadata={"lines": 1})

    assert call.name == "read_file"
    assert result.ok is True
    assert result.metadata == {"lines": 1}
