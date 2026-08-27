"""Safe, compact, real-time terminal rendering for tool and approval events."""

import re
from typing import Any

from rich.console import Console
from rich.text import Text

_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_SOURCE_LABELS = {
    "allow_all_mode": "allow-all mode",
    "execution_revalidation": "execution revalidation",
    "hard_policy": "hard safety policy",
    "human": "human",
    "human_timeout": "human approval timeout",
    "llm_reviewer": "LLM reviewer",
    "non_interactive_policy": "non-interactive policy",
    "prior_denial": "prior denial",
    "read_only_mode": "read-only mode",
    "safe_rule": "safe rule",
    "session_grant": "exact session grant",
}


class ConsoleEventReporter:
    """Render selected events without exposing tool arguments, contents, or raw output."""

    def __init__(self, console: Console, *, secrets: tuple[str, ...] = ()) -> None:
        self.console = console
        self._secrets = tuple(secret for secret in secrets if secret)

    def log(self, event_type: str, **data: Any) -> None:
        if event_type == "tool_call":
            self._tool_call(data)
        elif event_type == "approval_requested":
            self._approval_requested(data)
        elif event_type == "approval_decided":
            self._approval_decided(data)
        elif event_type == "tool_result":
            self._tool_result(data)

    def _tool_call(self, data: dict[str, Any]) -> None:
        detail = self._join(
            self._safe(data.get("name", "unknown tool")),
            f"step {data['step']}" if isinstance(data.get("step"), int) else None,
            self._short_call_id(data.get("tool_call_id")),
        )
        self._print("→", "Tool requested", detail, style="bold cyan")

    def _approval_requested(self, data: dict[str, Any]) -> None:
        risk = self._safe(data.get("risk", "unknown")).upper()
        tool = self._safe(data.get("tool", "unknown tool"))
        reasons = data.get("reasons")
        reason = None
        if isinstance(reasons, list):
            reason = "; ".join(self._safe(item, limit=100) for item in reasons[:2])
        target = None
        affected_paths = data.get("affected_paths")
        if isinstance(affected_paths, list) and affected_paths:
            target = f"target {self._safe(affected_paths[0], limit=120)}"
        program = data.get("program")
        command = None
        if isinstance(program, str) and program:
            argument_count = data.get("argument_count")
            suffix = ""
            if isinstance(argument_count, int):
                suffix = f" ({argument_count} {'arg' if argument_count == 1 else 'args'})"
            command = f"program {self._safe(program, limit=100)}{suffix}"
        detail = self._join(tool, f"risk {risk}", command, target, reason)
        style = {
            "SAFE": "green",
            "LOW": "green",
            "MEDIUM": "yellow",
            "HIGH": "bold red",
            "CRITICAL": "bold red",
        }.get(risk, "yellow")
        self._print("?", "Approval check", detail, style=style)

    def _approval_decided(self, data: dict[str, Any]) -> None:
        action = self._safe(data.get("action", "unknown")).lower()
        allowed = action == "allow"
        icon = "✓" if allowed else "✗"
        label = "Approval allowed" if allowed else "Approval denied"
        source_value = self._safe(data.get("source", "unknown"))
        source = _SOURCE_LABELS.get(source_value, source_value.replace("_", " "))
        detail = self._join(
            self._safe(data.get("tool", "unknown tool")),
            f"via {source}",
            self._scope(data.get("scope")),
            self._confidence(data.get("confidence")),
            self._decision_reason(data.get("reason")),
        )
        self._print(icon, label, detail, style="bold green" if allowed else "bold red")

    def _tool_result(self, data: dict[str, Any]) -> None:
        ok = data.get("ok") is True
        icon = "✓" if ok else "✗"
        label = "Tool completed" if ok else "Tool failed"
        metadata = data.get("metadata")
        exit_code = None
        duration = None
        if isinstance(metadata, dict):
            if isinstance(metadata.get("exit_code"), int):
                exit_code = f"exit {metadata['exit_code']}"
            if isinstance(metadata.get("duration_seconds"), (int, float)):
                duration = f"{metadata['duration_seconds']:g}s"
        target = None
        if isinstance(metadata, dict) and isinstance(metadata.get("path"), str):
            target = f"path {self._safe(metadata['path'], limit=120)}"
        error = None
        if not ok:
            error_value = data.get("error_code") or data.get("error")
            if error_value is not None:
                error = self._safe(error_value, limit=140)
        detail = self._join(
            self._safe(data.get("name", "unknown tool")),
            exit_code,
            duration,
            target,
            error,
            self._short_call_id(data.get("tool_call_id")),
        )
        self._print(icon, label, detail, style="bold green" if ok else "bold red")

    def _print(self, icon: str, label: str, detail: str, *, style: str) -> None:
        line = Text()
        line.append(f"{icon} {label}", style=style)
        if detail:
            line.append(f" · {detail}")
        self.console.print(line)

    def _safe(self, value: Any, *, limit: int = 180) -> str:
        text = str(value)
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        text = _INLINE_SECRET_PATTERN.sub(r"\1\2[REDACTED]", text)
        text = " ".join(text.split())
        if len(text) > limit:
            return f"{text[:limit]}…"
        return text

    @staticmethod
    def _join(*parts: str | None) -> str:
        return " · ".join(part for part in parts if part)

    @staticmethod
    def _short_call_id(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        return f"call {value[:12]}"

    @staticmethod
    def _scope(value: Any) -> str | None:
        if value == "session_exact":
            return "exact session scope"
        return None

    @staticmethod
    def _confidence(value: Any) -> str | None:
        if isinstance(value, (int, float)):
            return f"confidence {value:.2f}"
        return None

    def _decision_reason(self, value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        return self._safe(value, limit=120)
