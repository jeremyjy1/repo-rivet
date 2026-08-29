from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from repo_rivet.config import ApiConfig
from repo_rivet.llm.base import ModelContextLengthError, ModelRequestOptions
from repo_rivet.llm.openai_compatible import ModelRequestError, OpenAICompatibleClient
from repo_rivet.llm.protocol import InvalidConversationHistory


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        usage = SimpleNamespace(prompt_tokens=123, completion_tokens=7)
        return iter(
            [
                stream_chunk(content="done", finish_reason="stop"),
                SimpleNamespace(choices=[], usage=usage),
            ]
        )


class FragmentedToolCompletions:
    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        first_call = SimpleNamespace(
            index=0,
            id="call-1",
            type="function",
            function=SimpleNamespace(name="edit_", arguments='{"path":"app.py",'),
        )
        second_call = SimpleNamespace(
            index=0,
            id=None,
            type=None,
            function=SimpleNamespace(name="file", arguments='"operations":[]}'),
        )
        return iter(
            [
                stream_chunk(reasoning="inspect ", tool_calls=[first_call]),
                stream_chunk(reasoning="then edit", tool_calls=[second_call]),
                stream_chunk(finish_reason="tool_calls"),
            ]
        )


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def log(self, event_type: str, **data: object) -> None:
        self.events.append((event_type, data))


def stream_chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


class FailingCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        self.calls += 1
        raise self.error


class StructuredProviderError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("bad request")
        self.body = {"error": {"code": "context_length_exceeded"}}


class BadRequestProviderError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("invalid request api_key=test-key")
        self.status_code = 400
        self.code = "invalid_messages"
        self.request_id = "request-123"


class UnsupportedStreamOptionsError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Unknown parameter: stream_options")
        self.status_code = 400


class StreamOptionsFallbackCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.requests.append(kwargs)
        if "stream_options" in kwargs:
            raise UnsupportedStreamOptionsError()
        return iter([stream_chunk(content="fallback", finish_reason="stop")])


class APITimeoutError(RuntimeError):
    pass


class InterruptedStreamCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        self.calls += 1
        if self.calls == 1:
            return self._interrupted()
        return iter([stream_chunk(content="recovered", finish_reason="stop")])

    @staticmethod
    def _interrupted():  # type: ignore[no-untyped-def]
        yield stream_chunk(content="partial")
        raise APITimeoutError("stream stalled")


class StalledReasoningCompletions:
    def __init__(self, *, recover: bool = True) -> None:
        self.recover = recover
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.requests.append(kwargs)
        if len(self.requests) == 1 or not self.recover:
            return iter([stream_chunk(content=""), stream_chunk(reasoning="x" * 100)])
        return iter([stream_chunk(content="direct result", finish_reason="stop")])


class ReasoningProtocolError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "The `reasoning_content` in the thinking mode must be passed back to the API."
        )
        self.status_code = 400
        self.code = "invalid_request_error"


class ReasoningProtocolFallbackCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            raise ReasoningProtocolError()
        return iter([stream_chunk(content="protocol recovered", finish_reason="stop")])


def test_complete_calls_chat_completions_with_configured_model() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
    )
    adapter = OpenAICompatibleClient(config, client=client)

    result = adapter.complete(
        messages=[{"role": "user", "content": "task"}],
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )

    assert result.content == "done"
    assert result.input_tokens == 123
    assert result.output_tokens == 7
    assert completions.kwargs["model"] == "test-model"
    assert "max_tokens" not in completions.kwargs
    assert completions.kwargs["tool_choice"] == "auto"
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["stream_options"] == {"include_usage": True}


def test_complete_reassembles_streamed_reasoning_and_tool_arguments() -> None:
    completions = FragmentedToolCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = RecordingSink()
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
    )
    adapter = OpenAICompatibleClient(config, client=client, event_logger=events)

    result = adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    assert result.reasoning_content == "inspect then edit"
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].name == "edit_file"
    assert result.tool_calls[0].arguments == {"path": "app.py", "operations": []}
    progress = [data for name, data in events.events if name == "model_stream_progress"]
    assert progress
    assert progress[-1]["completed"] is True
    assert progress[-1]["tool_argument_chars"] > 0


def test_streaming_falls_back_when_provider_rejects_usage_options() -> None:
    completions = StreamOptionsFallbackCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = RecordingSink()
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
    )
    adapter = OpenAICompatibleClient(config, client=client, event_logger=events)

    result = adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    assert result.content == "fallback"
    assert len(completions.requests) == 2
    assert "stream_options" in completions.requests[0]
    assert "stream_options" not in completions.requests[1]
    assert any(name == "model_stream_usage_unavailable" for name, _data in events.events)


def test_interrupted_stream_retries_from_a_clean_accumulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = InterruptedStreamCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = RecordingSink()
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        max_retries=1,
    )
    monkeypatch.setattr("repo_rivet.llm.openai_compatible.time.sleep", lambda _delay: None)
    adapter = OpenAICompatibleClient(config, client=client, event_logger=events)

    result = adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    assert result.content == "recovered"
    assert "partial" not in result.content
    assert completions.calls == 2
    retry = next(data for name, data in events.events if name == "model_stream_retry")
    assert retry["attempt"] == 1
    assert retry["max_attempts"] == 2


def test_hidden_reasoning_stall_restarts_once_with_thinking_disabled() -> None:
    completions = StalledReasoningCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = RecordingSink()
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        max_retries=0,
        reasoning_stall_seconds=0,
        reasoning_stall_chars=100,
    )
    adapter = OpenAICompatibleClient(config, client=client, event_logger=events)

    result = adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    assert result.content == "direct result"
    assert result.provider_thinking_disabled is True
    assert result.reasoning_context_restart_required is True
    assert len(completions.requests) == 2
    assert "extra_body" not in completions.requests[0]
    assert completions.requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert all("max_tokens" not in request for request in completions.requests)
    recovery = next(
        data for name, data in events.events if name == "model_stream_reasoning_recovery"
    )
    assert recovery["reasoning_chars"] == 100
    assert recovery["recovery"] == "restart_with_thinking_disabled"


def test_hidden_reasoning_stall_recovery_is_bounded() -> None:
    completions = StalledReasoningCompletions(recover=False)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        max_retries=0,
        reasoning_stall_seconds=0,
        reasoning_stall_chars=100,
    )
    adapter = OpenAICompatibleClient(config, client=client)

    with pytest.raises(ModelRequestError, match="only hidden reasoning"):
        adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    assert len(completions.requests) == 2


def test_missing_reasoning_protocol_state_retries_once_without_thinking() -> None:
    completions = ReasoningProtocolFallbackCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = RecordingSink()
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        max_retries=0,
    )
    adapter = OpenAICompatibleClient(config, client=client, event_logger=events)

    result = adapter.complete(messages=[{"role": "user", "content": "continue"}], tools=[])

    assert result.content == "protocol recovered"
    assert result.provider_thinking_disabled is True
    assert result.reasoning_context_restart_required is True
    assert len(completions.requests) == 2
    assert completions.requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    recovery = next(
        data for name, data in events.events if name == "model_reasoning_protocol_recovery"
    )
    assert recovery["recovery"] == "retry_with_thinking_disabled"


def test_complete_applies_explicit_thinking_recovery_options() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
    )
    adapter = OpenAICompatibleClient(config, client=client)

    result = adapter.complete(
        messages=[{"role": "user", "content": "task"}],
        tools=[],
        options=ModelRequestOptions(thinking_enabled=False, reasoning_effort="low"),
    )

    assert result.provider_thinking_disabled is True
    assert result.reasoning_context_restart_required is False
    assert completions.kwargs["reasoning_effort"] == "low"
    assert completions.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize(
    "error",
    [
        StructuredProviderError(),
        RuntimeError("This model's maximum context length was exceeded"),
    ],
)
def test_context_overflow_is_normalized_for_controller_recovery(error: Exception) -> None:
    completions = FailingCompletions(error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
    )
    adapter = OpenAICompatibleClient(config, client=client)

    with pytest.raises(ModelContextLengthError):
        adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])


def test_non_retryable_bad_request_is_structured_redacted_and_not_retried() -> None:
    completions = FailingCompletions(BadRequestProviderError())
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        max_retries=3,
    )
    adapter = OpenAICompatibleClient(config, client=client)

    with pytest.raises(ModelRequestError) as raised:
        adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    record = raised.value.record
    assert completions.calls == 1
    assert record.status_code == 400
    assert record.error_code == "invalid_messages"
    assert record.request_id == "request-123"
    assert record.attempt == 1
    assert record.max_attempts == 4
    assert "test-key" not in record.message
    assert "[REDACTED]" in record.message


def test_invalid_tool_history_is_rejected_before_provider_call() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
    )
    adapter = OpenAICompatibleClient(config, client=client)

    with pytest.raises(InvalidConversationHistory, match="unfinished tool calls"):
        adapter.complete(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                }
            ],
            tools=[],
        )

    assert completions.kwargs is None
