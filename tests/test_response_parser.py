from types import SimpleNamespace

import pytest

from repo_rivet.llm.parser import ResponseParseError, ResponseParser


def response_with(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls=None,
    finish_reason="stop",
):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def function_call(*, arguments: str, name: str = "read_file", call_id: str = "call-1"):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, type="function", function=function)


def test_parse_text_and_function_call() -> None:
    parsed = ResponseParser().parse(
        response_with(
            content="I will inspect the file.",
            tool_calls=[function_call(arguments='{"path":"src/app.py"}')],
            finish_reason="tool_calls",
        )
    )

    assert parsed.content == "I will inspect the file."
    assert parsed.finish_reason == "tool_calls"
    assert parsed.tool_calls[0].arguments == {"path": "src/app.py"}


def test_reject_invalid_tool_json() -> None:
    with pytest.raises(ResponseParseError, match="invalid JSON"):
        ResponseParser().parse(response_with(tool_calls=[function_call(arguments="{")]))


def test_parse_provider_reasoning_content_for_continuation() -> None:
    parsed = ResponseParser().parse(
        response_with(reasoning_content="provider continuation state", finish_reason="length")
    )

    assert parsed.reasoning_content == "provider continuation state"
    assert parsed.as_assistant_message()["reasoning_content"] == "provider continuation state"


def test_reject_non_object_tool_arguments() -> None:
    with pytest.raises(ResponseParseError, match="JSON object"):
        ResponseParser().parse(response_with(tool_calls=[function_call(arguments="[]")]))


def test_reject_response_without_choices() -> None:
    with pytest.raises(ResponseParseError, match="no choices"):
        ResponseParser().parse(SimpleNamespace(choices=[]))
