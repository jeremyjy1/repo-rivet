"""OpenAI-compatible Chat Completions model adapter."""

import json
import re
import time
from dataclasses import replace
from typing import Any, cast

from openai import OpenAI

from repo_rivet.config import ApiConfig
from repo_rivet.llm.base import ModelContextLengthError, ModelRequestOptions, ModelResponse
from repo_rivet.llm.parser import ResponseParser
from repo_rivet.llm.protocol import find_pending_tool_calls, validate_tool_call_protocol
from repo_rivet.verification.models import ModelErrorRecord

_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)


class ModelRequestError(RuntimeError):
    """A sanitized, structured provider failure after explicit retry handling."""

    def __init__(self, record: ModelErrorRecord) -> None:
        super().__init__(record.message)
        self.record = record


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
            max_retries=0,
        )
        self._parser = parser or ResponseParser()

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: ModelRequestOptions | None = None,
    ) -> ModelResponse:
        validate_tool_call_protocol(messages)
        request_options = options or ModelRequestOptions()
        reasoning_effort = request_options.reasoning_effort or self._config.reasoning_effort
        thinking_enabled = request_options.thinking_enabled
        if thinking_enabled is None and self._config.thinking_mode != "provider_default":
            thinking_enabled = self._config.thinking_mode == "enabled"
        provider_options: dict[str, Any] = {}
        if reasoning_effort is not None:
            provider_options["reasoning_effort"] = reasoning_effort
        if thinking_enabled is not None:
            provider_options["extra_body"] = {
                "thinking": {"type": "enabled" if thinking_enabled else "disabled"}
            }
        max_attempts = self._config.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=cast(Any, messages),
                    tools=cast(Any, tools),
                    tool_choice="auto",
                    **provider_options,
                )
                break
            except Exception as error:
                if _is_context_length_error(error):
                    raise ModelContextLengthError("Provider context limit exceeded") from error
                retryable = _is_retryable_error(error)
                record = _model_error_record(
                    error,
                    messages=messages,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    retryable=retryable,
                    secrets=(self._config.api_key.get_secret_value(),),
                )
                if not retryable or attempt == max_attempts:
                    raise ModelRequestError(record) from error
                time.sleep(min(0.25 * (2 ** (attempt - 1)), 1.0))
        else:  # pragma: no cover - loop always returns or raises
            raise RuntimeError("model retry loop ended unexpectedly")
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


def _is_retryable_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or status_code >= 500
    return isinstance(error, (ConnectionError, TimeoutError)) or type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
    }


def _model_error_record(
    error: Exception,
    *,
    messages: list[dict[str, Any]],
    attempt: int,
    max_attempts: int,
    retryable: bool,
    secrets: tuple[str, ...],
) -> ModelErrorRecord:
    status_code = getattr(error, "status_code", None)
    error_code = getattr(error, "code", None)
    body = getattr(error, "body", None)
    if error_code is None and isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict):
            error_code = nested.get("code")
    request_id = getattr(error, "request_id", None)
    message = str(error)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = _INLINE_SECRET_PATTERN.sub(r"\1\2[REDACTED]", message)[:2_000]
    return ModelErrorRecord(
        error_type=type(error).__name__,
        status_code=status_code if isinstance(status_code, int) else None,
        error_code=str(error_code) if error_code is not None else None,
        request_id=str(request_id) if request_id is not None else None,
        message=message,
        retryable=retryable,
        attempt=attempt,
        max_attempts=max_attempts,
        message_count=len(messages),
        message_roles=[str(message.get("role", "unknown")) for message in messages],
        pending_tool_call_ids=find_pending_tool_calls(messages),
        request_size_bytes=len(
            json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8")
        ),
    )
