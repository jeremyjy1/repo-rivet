"""OpenAI-compatible Chat Completions model adapter."""

import json
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Protocol, cast

from openai import OpenAI

from repo_rivet.config import ApiConfig
from repo_rivet.llm.base import (
    REASONING_EFFORTS,
    ModelContextLengthError,
    ModelRequestOptions,
    ModelResponse,
    ModelStreamInterrupted,
    ReasoningEffort,
)
from repo_rivet.llm.parser import ResponseParseError, ResponseParser
from repo_rivet.llm.protocol import find_pending_tool_calls, validate_tool_call_protocol
from repo_rivet.reasoning.policy import clamp_effort, map_to_supported_effort
from repo_rivet.verification.models import ModelErrorRecord

_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_STREAM_PROGRESS_INTERVAL_SECONDS = 2.0


class ModelReasoningStalled(RuntimeError):
    """A stream produced private reasoning but no actionable response for too long."""


class ModelReasoningCeilingChanged(RuntimeError):
    """The user lowered the reasoning ceiling while a response was streaming."""

    def __init__(self, reasoning_effort: ReasoningEffort) -> None:
        super().__init__(reasoning_effort)
        self.reasoning_effort = reasoning_effort


class EventSink(Protocol):
    def log(self, event_type: str, **data: Any) -> None: ...


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
        event_logger: EventSink | None = None,
    ) -> None:
        self._config = config
        self._client = client or OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=str(config.base_url),
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self._parser = parser or ResponseParser()
        self._event_logger = event_logger
        self._interrupt_checker: Callable[[], bool] | None = None
        self._reasoning_effort_ceiling = config.reasoning_effort
        self._reasoning_supported_efforts = config.reasoning_supported_efforts
        self._reasoning_effort_lock = threading.Lock()

    def set_interrupt_checker(self, checker: Callable[[], bool] | None) -> None:
        """Install a cheap thread-safe check used to abandon an in-flight stream."""
        self._interrupt_checker = checker

    @property
    def reasoning_effort_ceiling(self) -> ReasoningEffort:
        """Expose the configured controller ceiling without leaking provider credentials."""
        with self._reasoning_effort_lock:
            return self._reasoning_effort_ceiling

    def set_reasoning_effort_ceiling(self, effort: str) -> None:
        """Update the ceiling safely; an active higher-effort stream restarts on its next chunk."""
        if effort not in REASONING_EFFORTS:
            raise ValueError(f"Unsupported reasoning effort: {effort}")
        with self._reasoning_effort_lock:
            self._reasoning_effort_ceiling = cast(ReasoningEffort, effort)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: ModelRequestOptions | None = None,
    ) -> ModelResponse:
        validate_tool_call_protocol(messages)
        request_options = options or ModelRequestOptions()
        requested_effort = request_options.reasoning_effort or self.reasoning_effort_ceiling
        requested_effort = _cap_reasoning_effort(requested_effort, self.reasoning_effort_ceiling)
        reasoning_effort = map_to_supported_effort(
            requested_effort, self._reasoning_supported_efforts
        )
        if reasoning_effort != requested_effort:
            self._log(
                "model_reasoning_effort_mapped",
                requested_effort=requested_effort,
                applied_effort=reasoning_effort,
                supported_efforts=list(self._reasoning_supported_efforts),
                reason="selected provider does not support the requested tier",
            )
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
        provider_thinking_disabled = thinking_enabled is False
        reasoning_context_restart_required = False
        max_attempts = self._config.max_retries + 1
        reasoning_protocol_recovery_used = False
        for attempt in range(1, max_attempts + 1):
            try:
                while True:
                    try:
                        self._raise_if_interrupted()
                        stream = self._create_stream(
                            messages=messages,
                            tools=tools,
                            provider_options=provider_options,
                            attempt=attempt,
                        )
                        parsed, usage = self._consume_stream(
                            stream,
                            attempt=attempt,
                            reasoning_effort=reasoning_effort,
                        )
                    except ModelReasoningCeilingChanged as error:
                        previous_effort = reasoning_effort
                        reasoning_effort = error.reasoning_effort
                        provider_options["reasoning_effort"] = reasoning_effort
                        self._log(
                            "model_reasoning_effort_downgraded",
                            attempt=attempt,
                            previous_effort=previous_effort,
                            reasoning_effort=reasoning_effort,
                            elapsed_seconds=None,
                            reason="reasoning ceiling changed during active stream",
                        )
                        continue
                    except ModelReasoningStalled as error:
                        lower_effort = _lower_reasoning_effort(reasoning_effort)
                        if lower_effort is None:
                            raise
                        lower_effort = map_to_supported_effort(
                            lower_effort,
                            self._reasoning_supported_efforts,
                        )
                        if lower_effort == reasoning_effort:
                            raise
                        previous_effort = reasoning_effort
                        reasoning_effort = lower_effort
                        provider_options["reasoning_effort"] = lower_effort
                        self._log(
                            "model_reasoning_effort_downgraded",
                            attempt=attempt,
                            previous_effort=previous_effort,
                            reasoning_effort=lower_effort,
                            elapsed_seconds=error.args[0],
                            reason="no actionable stream output",
                        )
                        continue
                    except ModelStreamInterrupted:
                        raise
                    except Exception as error:
                        if (
                            not _reasoning_content_replay_required(error)
                            or reasoning_protocol_recovery_used
                        ):
                            raise
                        reasoning_protocol_recovery_used = True
                        provider_options = _disable_provider_thinking(provider_options)
                        provider_thinking_disabled = True
                        reasoning_context_restart_required = True
                        self._log(
                            "model_reasoning_protocol_recovery",
                            attempt=attempt,
                            error_type=type(error).__name__,
                            recovery="retry_with_thinking_disabled",
                        )
                        continue
                    break
                break
            except ModelStreamInterrupted:
                raise
            except ResponseParseError:
                raise
            except Exception as error:
                if self._interrupted():
                    raise ModelStreamInterrupted("model stream redirected by user") from error
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
                delay = min(0.25 * (2 ** (attempt - 1)), 1.0)
                self._log(
                    "model_stream_retry",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error_type=type(error).__name__,
                    delay_seconds=delay,
                )
                time.sleep(delay)
        else:  # pragma: no cover - loop always returns or raises
            raise RuntimeError("model retry loop ended unexpectedly")
        return replace(
            parsed,
            input_tokens=_usage_value(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_usage_value(usage, "completion_tokens", "output_tokens"),
            provider_thinking_disabled=provider_thinking_disabled,
            reasoning_context_restart_required=reasoning_context_restart_required,
        )

    def _create_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        provider_options: dict[str, Any],
        attempt: int,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self._config.model,
            "messages": cast(Any, messages),
            "tools": cast(Any, tools),
            "tool_choice": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            **provider_options,
        }
        try:
            return self._client.chat.completions.create(**request)
        except Exception as error:
            if not _stream_options_are_unsupported(error):
                raise
            request.pop("stream_options")
            self._log(
                "model_stream_usage_unavailable",
                attempt=attempt,
                reason="provider rejected stream_options",
            )
            return self._client.chat.completions.create(**request)

    def _consume_stream(
        self,
        stream: Any,
        *,
        attempt: int,
        reasoning_effort: str | None,
    ) -> tuple[ModelResponse, Any]:
        started_at = time.monotonic()
        last_progress_at = started_at
        chunk_count = 0
        content_parts: list[str] = []
        content_chars = 0
        reasoning_parts: list[str] = []
        reasoning_chars = 0
        tool_fragments: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: Any = None

        for chunk in _iter_stream_and_close(stream):
            self._raise_if_interrupted()
            chunk_count += 1
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            choices = getattr(chunk, "choices", None) or []
            if choices:
                choice = choices[0]
                finish = getattr(choice, "finish_reason", None)
                if isinstance(finish, str):
                    finish_reason = finish
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    content = getattr(delta, "content", None)
                    if content is not None:
                        if not isinstance(content, str):
                            raise ResponseParseError("Streamed model response content is not text")
                        content_parts.append(content)
                        content_chars += len(content)
                    reasoning = _stream_reasoning_text(delta)
                    if reasoning is not None:
                        reasoning_parts.append(reasoning)
                        reasoning_chars += len(reasoning)
                    for fallback_index, raw_call in enumerate(
                        getattr(delta, "tool_calls", None) or []
                    ):
                        _append_tool_call_fragment(
                            tool_fragments,
                            raw_call,
                            fallback_index=fallback_index,
                        )

            now = time.monotonic()
            elapsed_seconds = now - started_at
            tool_argument_chars = sum(
                len("".join(fragment["arguments"])) for fragment in tool_fragments.values()
            )
            current_ceiling = self.reasoning_effort_ceiling
            bounded_effort = map_to_supported_effort(
                _cap_reasoning_effort(reasoning_effort, current_ceiling),
                self._reasoning_supported_efforts,
            )
            if bounded_effort != reasoning_effort and not content_chars and not tool_argument_chars:
                raise ModelReasoningCeilingChanged(bounded_effort)
            if (
                reasoning_chars
                and not content_chars
                and not tool_argument_chars
                and reasoning_effort in {"high", "xhigh", "max"}
                and elapsed_seconds >= self._config.reasoning_stall_seconds
            ):
                raise ModelReasoningStalled(round(elapsed_seconds, 1))
            if chunk_count == 1 or now - last_progress_at >= _STREAM_PROGRESS_INTERVAL_SECONDS:
                self._log_stream_progress(
                    attempt=attempt,
                    started_at=started_at,
                    chunk_count=chunk_count,
                    content_parts=content_parts,
                    reasoning_parts=reasoning_parts,
                    tool_fragments=tool_fragments,
                    reasoning_effort=reasoning_effort,
                )
                last_progress_at = now

        self._raise_if_interrupted()
        if chunk_count == 0:
            raise ResponseParseError("Model stream contained no chunks")
        self._log_stream_progress(
            attempt=attempt,
            started_at=started_at,
            chunk_count=chunk_count,
            content_parts=content_parts,
            reasoning_parts=reasoning_parts,
            tool_fragments=tool_fragments,
            reasoning_effort=reasoning_effort,
            completed=True,
        )
        response = _assembled_stream_response(
            content_parts=content_parts,
            reasoning_parts=reasoning_parts,
            tool_fragments=tool_fragments,
            finish_reason=finish_reason,
        )
        return self._parser.parse(response), usage

    def _raise_if_interrupted(self) -> None:
        if self._interrupted():
            raise ModelStreamInterrupted("model stream redirected by user")

    def _interrupted(self) -> bool:
        checker = self._interrupt_checker
        return checker is not None and checker()

    def _log_stream_progress(
        self,
        *,
        attempt: int,
        started_at: float,
        chunk_count: int,
        content_parts: list[str],
        reasoning_parts: list[str],
        tool_fragments: dict[int, dict[str, Any]],
        reasoning_effort: str | None,
        completed: bool = False,
    ) -> None:
        content_chars = sum(len(part) for part in content_parts)
        reasoning_chars = sum(len(part) for part in reasoning_parts)
        tool_argument_chars = sum(
            len("".join(fragment["arguments"])) for fragment in tool_fragments.values()
        )
        self._log(
            "model_stream_progress",
            attempt=attempt,
            elapsed_seconds=round(time.monotonic() - started_at, 1),
            chunk_count=chunk_count,
            content_chars=content_chars,
            reasoning_chars=reasoning_chars,
            tool_argument_chars=tool_argument_chars,
            activity_phase=_stream_activity_phase(
                content_chars=content_chars,
                reasoning_chars=reasoning_chars,
                tool_argument_chars=tool_argument_chars,
                completed=completed,
            ),
            reasoning_effort=reasoning_effort,
            completed=completed,
        )

    def _log(self, event_type: str, **data: Any) -> None:
        if self._event_logger is not None:
            self._event_logger.log(event_type, **data)


def _stream_activity_phase(
    *,
    content_chars: int,
    reasoning_chars: int,
    tool_argument_chars: int,
    completed: bool,
) -> str:
    """Describe stream progress without exposing provider-private reasoning text."""
    if completed:
        return "completed"
    if tool_argument_chars:
        return "preparing_tool"
    if content_chars:
        return "composing_answer"
    if reasoning_chars >= 20_000:
        return "refining_action"
    if reasoning_chars >= 8_000:
        return "evaluating_options"
    if reasoning_chars >= 2_000:
        return "analyzing_context"
    if reasoning_chars:
        return "understanding_task"
    return "waiting"


def _lower_reasoning_effort(effort: str | None) -> ReasoningEffort | None:
    if effort not in REASONING_EFFORTS:
        return None
    index = REASONING_EFFORTS.index(cast(ReasoningEffort, effort))
    return REASONING_EFFORTS[index - 1] if index else None


def _cap_reasoning_effort(
    effort: ReasoningEffort,
    ceiling: ReasoningEffort,
) -> ReasoningEffort:
    return clamp_effort(effort, "low", ceiling)


def _stream_reasoning_text(delta: Any) -> str | None:
    value = getattr(delta, "reasoning_content", None)
    if value is None:
        value = getattr(delta, "reasoning", None)
    if value is None:
        model_extra = getattr(delta, "model_extra", None)
        if isinstance(model_extra, dict):
            value = model_extra.get("reasoning_content", model_extra.get("reasoning"))
    if value is not None and not isinstance(value, str):
        raise ResponseParseError("Streamed model response reasoning_content is not text")
    return value


def _iter_stream_and_close(stream: Any) -> Iterator[Any]:
    try:
        yield from stream
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def _append_tool_call_fragment(
    fragments: dict[int, dict[str, Any]],
    raw_call: Any,
    *,
    fallback_index: int,
) -> None:
    index = getattr(raw_call, "index", fallback_index)
    if not isinstance(index, int) or index < 0:
        raise ResponseParseError("Streamed tool call has an invalid index")
    fragment = fragments.setdefault(
        index,
        {"id": None, "type": "function", "name": [], "arguments": []},
    )
    call_id = getattr(raw_call, "id", None)
    if isinstance(call_id, str) and call_id:
        existing_id = fragment["id"]
        if existing_id is not None and existing_id != call_id:
            raise ResponseParseError("Streamed tool call changed its id")
        fragment["id"] = call_id
    call_type = getattr(raw_call, "type", None)
    if isinstance(call_type, str) and call_type:
        fragment["type"] = call_type
    function = getattr(raw_call, "function", None)
    if function is None:
        return
    name = getattr(function, "name", None)
    if isinstance(name, str) and name:
        fragment["name"].append(name)
    arguments = getattr(function, "arguments", None)
    if isinstance(arguments, str) and arguments:
        fragment["arguments"].append(arguments)


def _assembled_stream_response(
    *,
    content_parts: list[str],
    reasoning_parts: list[str],
    tool_fragments: dict[int, dict[str, Any]],
    finish_reason: str | None,
) -> Any:
    tool_calls = []
    for index in sorted(tool_fragments):
        fragment = tool_fragments[index]
        tool_calls.append(
            SimpleNamespace(
                id=fragment["id"],
                type=fragment["type"],
                function=SimpleNamespace(
                    name="".join(fragment["name"]),
                    arguments="".join(fragment["arguments"]),
                ),
            )
        )
    message = SimpleNamespace(
        content="".join(content_parts) or None,
        reasoning_content="".join(reasoning_parts) or None,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
    )


def _stream_options_are_unsupported(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code not in {400, 422}:
        return False
    message = str(error).lower()
    body = getattr(error, "body", None)
    if body is not None:
        message += " " + str(body).lower()
    return "stream_options" in message or "stream options" in message


def _reasoning_content_replay_required(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "reasoning_content" in message
        and "thinking mode" in message
        and ("passed back" in message or "required" in message)
    )


def _disable_provider_thinking(provider_options: dict[str, Any]) -> dict[str, Any]:
    """Return a clean retry configuration that asks compatible providers for direct output."""
    recovered = dict(provider_options)
    recovered.pop("reasoning_effort", None)
    extra_body = recovered.get("extra_body")
    recovered_extra = dict(extra_body) if isinstance(extra_body, dict) else {}
    recovered_extra["thinking"] = {"type": "disabled"}
    recovered["extra_body"] = recovered_extra
    return recovered


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
