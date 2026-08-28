"""Local validation for OpenAI-compatible assistant/tool message pairing."""

from __future__ import annotations

import re
from typing import Any

_EMBEDDED_TOOL_PROTOCOL = re.compile(
    r"<\|{1,2}DSML\|{1,2}(?:tool_calls|invoke)\b",
    flags=re.IGNORECASE,
)


class InvalidConversationHistory(ValueError):
    """Conversation messages violate the tool-call/result protocol."""


def contains_embedded_tool_protocol(text: str | None) -> bool:
    """Detect provider tool markup leaked into the assistant text channel."""
    if not text:
        return False
    normalized = text.replace("｜", "|").replace("\\", "")
    return _EMBEDDED_TOOL_PROTOCOL.search(normalized) is not None


def validate_tool_call_protocol(messages: list[dict[str, Any]]) -> None:
    pending: dict[str, str] = {}
    seen: set[str] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            if pending:
                raise InvalidConversationHistory(
                    f"message {index} starts before tool results were returned for: "
                    f"{', '.join(pending)}"
                )
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise InvalidConversationHistory(f"message {index} has invalid tool_calls")
            for raw_call in calls:
                if not isinstance(raw_call, dict):
                    raise InvalidConversationHistory(f"message {index} has an invalid tool call")
                call_id = raw_call.get("id")
                function = raw_call.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    raise InvalidConversationHistory(f"message {index} has a missing tool call ID")
                if call_id in seen:
                    raise InvalidConversationHistory(f"duplicate tool call ID: {call_id}")
                pending[call_id] = str(name or "unknown")
                seen.add(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise InvalidConversationHistory(
                    f"message {index} has an unknown tool result ID: {call_id}"
                )
            del pending[call_id]
        elif pending:
            raise InvalidConversationHistory(
                f"message {index} appears before tool results were returned for: "
                f"{', '.join(pending)}"
            )
    if pending:
        raise InvalidConversationHistory(
            "conversation ends with unfinished tool calls: " + ", ".join(pending)
        )


def find_pending_tool_calls(messages: list[dict[str, Any]]) -> list[str]:
    pending: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "assistant":
            for raw_call in message.get("tool_calls") or []:
                if isinstance(raw_call, dict) and isinstance(raw_call.get("id"), str):
                    pending[raw_call["id"]] = str(raw_call.get("id"))
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str):
                pending.pop(call_id, None)
    return list(pending)
