from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from repo_rivet.config import ApiConfig
from repo_rivet.llm.base import (
    ModelContextLengthError,
    ModelRequestOptions,
    ModelStreamInterrupted,
)
from repo_rivet.llm.openai_compatible import (
    ModelRequestError,
    OpenAICompatibleClient,
    _stream_activity_phase,
)
from repo_rivet.llm.protocol import InvalidConversationHistory


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
        self.kwargs = kwargs
        usage = SimpleNamespace(prompt_tokens=123, completion_tokens=7)
        return iter(
            [
                stream_chunk(content="done", finish_reason="stop"),
                SimpleNamespace(choices=[], usage=usage),
            ]
        )


class FragmentedToolCompletions:
    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
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


class StallingThenCompletes:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return iter([stream_chunk(reasoning="still analyzing")])
        return iter([stream_chunk(content="done", finish_reason="stop")])


class ReconfiguredThenCompletes:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.on_first_chunk = lambda: None

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
        self.requests.append(kwargs)
        if len(self.requests) > 1:
            return iter([stream_chunk(content="done", finish_reason="stop")])

        def first_stream() -> Iterator[SimpleNamespace]:
            self.on_first_chunk()
            yield stream_chunk(reasoning="started at the old ceiling")

        return first_stream()


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

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
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

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
        self.requests.append(kwargs)
        if "stream_options" in kwargs:
            raise UnsupportedStreamOptionsError()
        return iter([stream_chunk(content="fallback", finish_reason="stop")])


class APITimeoutError(RuntimeError):
    pass


class InterruptedStreamCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
        del kwargs
        self.calls += 1
        if self.calls == 1:
            return self._interrupted()
        return iter([stream_chunk(content="recovered", finish_reason="stop")])

    @staticmethod
    def _interrupted() -> Iterator[SimpleNamespace]:
        yield stream_chunk(content="partial")
        raise APITimeoutError("stream stalled")


class RedirectedStreamCompletions:
    def __init__(self, redirected: dict[str, bool]) -> None:
        self.redirected = redirected
        self.calls = 0

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
        del kwargs
        self.calls += 1
        return self._stream()

    def _stream(self) -> Iterator[SimpleNamespace]:
        yield stream_chunk(content="discarded partial response")
        self.redirected["value"] = True
        yield stream_chunk(content="must not be consumed", finish_reason="stop")


class ContinuousReasoningCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
        self.requests.append(kwargs)
        return iter(
            [
                stream_chunk(reasoning="x" * 25_000),
                stream_chunk(content="considered result", finish_reason="stop"),
            ]
        )


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

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            raise ReasoningProtocolError()
        return iter([stream_chunk(content="protocol recovered", finish_reason="stop")])


class ThinkingToolChoiceError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Thinking mode does not support this tool_choice")
        self.status_code = 400
        self.code = "invalid_request_error"


class ThinkingToolChoiceFallbackCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> Iterator[SimpleNamespace]:
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            raise ThinkingToolChoiceError()
        return iter([stream_chunk(content="forced tool recovered", finish_reason="stop")])


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


def test_complete_can_require_the_only_recovery_tool() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
    )
    adapter = OpenAICompatibleClient(config, client=client)

    adapter.complete(
        messages=[{"role": "user", "content": "register checks"}],
        tools=[{"type": "function", "function": {"name": "register_verification"}}],
        options=ModelRequestOptions(required_tool="register_verification"),
    )

    assert completions.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "register_verification"},
    }


def test_required_tool_retries_without_thinking_when_provider_rejects_combination() -> None:
    completions = ThinkingToolChoiceFallbackCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = RecordingSink()
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        reasoning_effort="low",
    )
    adapter = OpenAICompatibleClient(config, client=client, event_logger=events)

    result = adapter.complete(
        messages=[{"role": "user", "content": "register checks"}],
        tools=[{"type": "function", "function": {"name": "register_verification"}}],
        options=ModelRequestOptions(required_tool="register_verification"),
    )

    assert result.content == "forced tool recovered"
    assert result.provider_thinking_disabled is True
    assert len(completions.requests) == 2
    assert completions.requests[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in completions.requests[1]
    assert completions.requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert any(name == "model_thinking_tool_choice_recovery" for name, _data in events.events)


def test_complete_maps_adaptive_effort_to_provider_capabilities() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = RecordingSink()
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        reasoning_effort="max",
        reasoning_supported_efforts=("low", "high", "max"),
    )
    adapter = OpenAICompatibleClient(config, client=client, event_logger=events)

    adapter.complete(
        messages=[{"role": "user", "content": "task"}],
        tools=[],
        options=ModelRequestOptions(reasoning_effort="xhigh"),
    )

    assert completions.kwargs["reasoning_effort"] == "high"
    mapped = next(data for name, data in events.events if name == "model_reasoning_effort_mapped")
    assert mapped["requested_effort"] == "xhigh"
    assert mapped["applied_effort"] == "high"


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


def test_reasoning_only_stream_downgrades_effort_and_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = StallingThenCompletes()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = RecordingSink()
    ticks = iter(range(0, 1_000, 100))
    monkeypatch.setattr(
        "repo_rivet.llm.openai_compatible.time.monotonic",
        lambda: next(ticks),
    )
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        reasoning_effort="max",
        reasoning_stall_seconds=45,
    )
    adapter = OpenAICompatibleClient(config, client=client, event_logger=events)

    result = adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    assert result.content == "done"
    assert [request["reasoning_effort"] for request in completions.requests] == [
        "max",
        "xhigh",
    ]
    downgrade = next(
        data for name, data in events.events if name == "model_reasoning_effort_downgraded"
    )
    assert downgrade["previous_effort"] == "max"
    assert downgrade["reasoning_effort"] == "xhigh"
    assert downgrade["elapsed_seconds"] == 100


def test_lowering_live_reasoning_ceiling_restarts_unfinished_stream() -> None:
    completions = ReconfiguredThenCompletes()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = RecordingSink()
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        reasoning_effort="max",
    )
    adapter = OpenAICompatibleClient(config, client=client, event_logger=events)
    completions.on_first_chunk = lambda: adapter.set_reasoning_effort_ceiling("low")

    result = adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    assert result.content == "done"
    assert [request["reasoning_effort"] for request in completions.requests] == [
        "max",
        "low",
    ]
    downgrade = next(
        data for name, data in events.events if name == "model_reasoning_effort_downgraded"
    )
    assert downgrade["reason"] == "reasoning ceiling changed during active stream"
    assert downgrade["previous_effort"] == "max"
    assert downgrade["reasoning_effort"] == "low"


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

    tools = [{"type": "function", "function": {"name": "register_verification"}}]
    result = adapter.complete(
        messages=[{"role": "user", "content": "task"}],
        tools=tools,
        options=ModelRequestOptions(required_tool="register_verification"),
    )

    assert result.content == "fallback"
    assert len(completions.requests) == 2
    assert "stream_options" in completions.requests[0]
    assert "stream_options" not in completions.requests[1]
    assert all(
        request["tool_choice"]
        == {
            "type": "function",
            "function": {"name": "register_verification"},
        }
        for request in completions.requests
    )
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


def test_user_redirect_abandons_stream_without_provider_retry() -> None:
    redirected = {"value": False}
    completions = RedirectedStreamCompletions(redirected)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        max_retries=2,
    )
    adapter = OpenAICompatibleClient(config, client=client)
    adapter.set_interrupt_checker(lambda: redirected["value"])

    with pytest.raises(ModelStreamInterrupted):
        adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    assert completions.calls == 1


def test_continuous_hidden_reasoning_is_allowed_to_finish() -> None:
    completions = ContinuousReasoningCompletions()
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

    result = adapter.complete(messages=[{"role": "user", "content": "task"}], tools=[])

    assert result.content == "considered result"
    assert result.reasoning_content == "x" * 25_000
    assert result.provider_thinking_disabled is False
    assert result.reasoning_context_restart_required is False
    assert len(completions.requests) == 1
    assert any(name == "model_stream_progress" for name, _ in events.events)


@pytest.mark.parametrize(
    ("reasoning_chars", "expected"),
    [
        (0, "waiting"),
        (1, "understanding_task"),
        (2_000, "analyzing_context"),
        (8_000, "evaluating_options"),
        (20_000, "refining_action"),
    ],
)
def test_stream_activity_phase_describes_reasoning_progress(
    reasoning_chars: int,
    expected: str,
) -> None:
    assert (
        _stream_activity_phase(
            content_chars=0,
            reasoning_chars=reasoning_chars,
            tool_argument_chars=0,
            completed=False,
        )
        == expected
    )


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
