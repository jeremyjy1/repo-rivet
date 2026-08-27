"""OpenAI-compatible Chat Completions model adapter."""

from dataclasses import replace
from typing import Any, cast

from openai import OpenAI

from repo_rivet.config import ApiConfig
from repo_rivet.llm.base import ModelContextLengthError, ModelResponse
from repo_rivet.llm.parser import ResponseParser


class OpenAICompatibleClient:
    """Call a configured Chat Completions endpoint using native tool calling."""

    def __init__(
        self,
        config: ApiConfig,
        *,
        client: Any | None = None,
        parser: ResponseParser | None = None,
    ) -> None:
        self._config = config
        self._client = client or OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=str(config.base_url),
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self._parser = parser or ResponseParser()

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=cast(Any, messages),
                tools=cast(Any, tools),
                tool_choice="auto",
                max_tokens=self._config.max_output_tokens,
            )
        except Exception as error:
            if _is_context_length_error(error):
                raise ModelContextLengthError("Provider context limit exceeded") from error
            raise
        parsed = self._parser.parse(response)
        usage = getattr(response, "usage", None)
        return replace(
            parsed,
            input_tokens=_usage_value(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_usage_value(usage, "completion_tokens", "output_tokens"),
        )


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(usage, name, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _is_context_length_error(error: Exception) -> bool:
    markers = {
        "context_length_exceeded",
        "context_window_exceeded",
        "max_tokens_exceeded",
    }
    structured_values: list[str] = []
    for attribute in ("code", "type"):
        value = getattr(error, attribute, None)
        if isinstance(value, str):
            structured_values.append(value.lower())
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict):
            for key in ("code", "type"):
                value = nested.get(key)
                if isinstance(value, str):
                    structured_values.append(value.lower())
    if any(value in markers for value in structured_values):
        return True

    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "maximum context length",
            "context length exceeded",
            "prompt too long",
            "input too large",
            "token limit exceeded",
        )
    )
