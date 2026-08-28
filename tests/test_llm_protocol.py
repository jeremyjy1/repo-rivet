import pytest

from repo_rivet.llm.protocol import (
    InvalidConversationHistory,
    contains_embedded_tool_protocol,
    validate_tool_call_protocol,
)


def assistant_calls(*call_ids: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    }


def test_multiple_tool_calls_require_and_accept_every_matching_result() -> None:
    validate_tool_call_protocol(
        [
            assistant_calls("call-1", "call-2"),
            {"role": "tool", "tool_call_id": "call-1", "content": "one"},
            {"role": "tool", "tool_call_id": "call-2", "content": "two"},
            {"role": "assistant", "content": "done"},
        ]
    )


def test_unknown_tool_result_is_rejected() -> None:
    with pytest.raises(InvalidConversationHistory, match="unknown tool result ID"):
        validate_tool_call_protocol(
            [{"role": "tool", "tool_call_id": "unknown", "content": "result"}]
        )


def test_message_between_tool_call_and_result_is_rejected() -> None:
    with pytest.raises(InvalidConversationHistory, match="before tool results"):
        validate_tool_call_protocol(
            [assistant_calls("call-1"), {"role": "system", "content": "interrupt"}]
        )


def test_duplicate_tool_call_id_is_rejected() -> None:
    with pytest.raises(InvalidConversationHistory, match="duplicate tool call ID"):
        validate_tool_call_protocol(
            [
                assistant_calls("call-1"),
                {"role": "tool", "tool_call_id": "call-1", "content": "one"},
                assistant_calls("call-1"),
            ]
        )


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_assistant_message_is_rejected(content: str | None) -> None:
    with pytest.raises(InvalidConversationHistory, match="neither assistant content"):
        validate_tool_call_protocol([{"role": "assistant", "content": content}])


def test_detect_dsml_tool_markup_leaked_into_text() -> None:
    assert contains_embedded_tool_protocol(
        'Build passed.\n<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="run_verification">'
    )
    assert not contains_embedded_tool_protocol("Build and self-test passed.")
