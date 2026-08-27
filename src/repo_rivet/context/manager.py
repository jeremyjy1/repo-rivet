"""Build bounded model context from task, state, and conversation history."""

from copy import deepcopy
from typing import Any

SYSTEM_PROMPT = """You are RepoRivet, a local coding agent.
Work only through the provided tools and stay inside the configured workspace.
Inspect relevant files before editing. Prefer precise replacements over full rewrites.
Treat command failures as observations, diagnose them, and continue when possible.
After changing files, run an appropriate test, build, lint, or syntax check before finishing.
Do not claim success unless the latest verification after the latest change succeeded.
When finished, summarize the changes and verification concisely."""


class ContextManager:
    """Keep recent messages verbatim and compress older history deterministically."""

    def __init__(
        self,
        *,
        recent_message_limit: int = 16,
        max_tool_content_chars: int = 20_000,
        max_summary_chars: int = 4_000,
    ) -> None:
        if recent_message_limit <= 0:
            raise ValueError("recent_message_limit must be positive")
        if max_tool_content_chars <= 0 or max_summary_chars <= 0:
            raise ValueError("context size limits must be positive")
        self._recent_message_limit = recent_message_limit
        self._max_tool_content_chars = max_tool_content_chars
        self._max_summary_chars = max_summary_chars

    def build(
        self,
        *,
        task: str,
        history: list[dict[str, Any]],
        state_summary: str,
        remaining_steps: int,
    ) -> list[dict[str, Any]]:
        """Construct the messages for one model call without mutating history."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        older = history[: -self._recent_message_limit]
        recent = history[-self._recent_message_limit :]
        if older:
            messages.append(
                {
                    "role": "system",
                    "content": f"Earlier context summary:\n{self._summarize(older)}",
                }
            )

        messages.extend(self._bounded_message(message) for message in recent)
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Current state:\n{state_summary}\n"
                    f"Remaining agent steps: {max(remaining_steps, 0)}"
                ),
            }
        )
        return messages

    def _bounded_message(self, message: dict[str, Any]) -> dict[str, Any]:
        bounded = deepcopy(message)
        content = bounded.get("content")
        if bounded.get("role") == "tool" and isinstance(content, str):
            bounded["content"] = self._truncate(content, self._max_tool_content_chars)
        return bounded

    def _summarize(self, history: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in history:
            role = str(message.get("role", "unknown"))
            content = message.get("content")
            if isinstance(content, str) and content:
                compact = " ".join(content.split())
                lines.append(f"- {role}: {compact[:500]}")
            elif message.get("tool_calls"):
                names = [
                    str(call.get("function", {}).get("name", "unknown"))
                    for call in message["tool_calls"]
                    if isinstance(call, dict)
                ]
                lines.append(f"- {role}: requested tools {', '.join(names)}")
        summary = "\n".join(lines) or "(no textual history)"
        return self._truncate(summary, self._max_summary_chars)

    @staticmethod
    def _truncate(content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        marker = "\n... content truncated ...\n"
        side = (limit - len(marker)) // 2
        return f"{content[:side]}{marker}{content[-side:]}"
