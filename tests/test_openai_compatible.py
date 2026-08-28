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
        message = SimpleNamespace(content="done", tool_calls=None)
        usage = SimpleNamespace(prompt_tokens=123, completion_tokens=7)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=usage,
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

    adapter.complete(
        messages=[{"role": "user", "content": "task"}],
        tools=[],
        options=ModelRequestOptions(thinking_enabled=False, reasoning_effort="low"),
    )

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
