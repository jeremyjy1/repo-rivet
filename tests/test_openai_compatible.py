from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from repo_rivet.config import ApiConfig
from repo_rivet.llm.base import ModelContextLengthError
from repo_rivet.llm.openai_compatible import OpenAICompatibleClient


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

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise self.error


class StructuredProviderError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("bad request")
        self.body = {"error": {"code": "context_length_exceeded"}}


def test_complete_calls_chat_completions_with_configured_model() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = ApiConfig(
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="test-model",
        context_window_tokens=32768,
        max_output_tokens=2048,
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
    assert completions.kwargs["max_tokens"] == 2048
    assert completions.kwargs["tool_choice"] == "auto"


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
