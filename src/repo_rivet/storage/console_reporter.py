"""Safe, compact, real-time terminal rendering for tool and approval events."""

import re
from typing import Any

from rich.console import Console
from rich.status import Status
from rich.text import Text

from repo_rivet.reasoning.models import ReasoningDisplayMode
from repo_rivet.storage.terminal_text import escape_terminal_controls

_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_INTERNAL_EVENT_ID_PATTERN = re.compile(r"\b(?:obs|reason)-[A-Za-z0-9_-]+\b")
_SOURCE_LABELS = {
    "allow_all_mode": "allow-all mode",
    "execution_revalidation": "execution revalidation",
    "hard_policy": "hard safety policy",
    "human": "human",
    "human_timeout": "human approval timeout",
    "llm_reviewer": "LLM reviewer",
    "non_interactive_policy": "non-interactive policy",
    "prior_denial": "prior denial",
    "safe_rule": "safe rule",
    "session_grant": "exact session grant",
}
_QUIET_ALLOW_SOURCES = {
    "allow_all_mode",
    "safe_rule",
    "session_grant",
}
_APPROVAL_FAILURE_CODES = {
    "approval_denied",
    "approval_stale",
    "hard_policy_denied",
}
_TOOL_PROGRESS_LABELS = {
    "edit_file": "applying edits",
    "git_diff": "inspecting Git changes",
    "git_status": "inspecting Git status",
    "list_files": "listing workspace files",
    "read_file": "reading file",
    "run_command": "running command",
    "run_verification": "running verification",
    "search_text": "searching workspace text",
    "write_file": "creating file",
}
_STATUS_STOP_EVENTS = {
    "approved_tool_executed",
    "approval_awaiting_human",
    "llm_approval_review_failed",
    "llm_approval_reviewed",
    "model_call_finished",
    "session_end",
    "tool_result",
}


class ConsoleEventReporter:
    """Render selected events without exposing tool arguments, contents, or raw output."""

    def __init__(
        self,
        console: Console,
        *,
        secrets: tuple[str, ...] = (),
        reasoning_mode: ReasoningDisplayMode = ReasoningDisplayMode.OFF,
    ) -> None:
        self.console = console
        self._secrets = tuple(secret for secret in secrets if secret)
        self.reasoning_mode = reasoning_mode
        self._active_status: Status | None = None

    def log(self, event_type: str, **data: Any) -> None:
        if event_type in _STATUS_STOP_EVENTS:
            self._stop_status()

        if event_type == "model_call":
            self._start_model_status()
        elif event_type == "llm_approval_review_started":
            tool = self._safe(data.get("tool", "tool"), limit=80)
            self._start_status(f"[cyan]Approval model is reviewing {tool}…[/cyan]")
        elif event_type == "approval_decided":
            self._approval_decided(data)
        elif event_type == "approved_tool_started":
            self._tool_started(data)
        elif event_type == "reasoning":
            self._reasoning(data)
        elif event_type == "assessment":
            self._assessment(data)
        elif event_type == "action":
            self._action(data)
        elif event_type == "action_blocked":
            self._action_blocked(data)
        elif event_type == "observation":
            self._observation(data)
        elif event_type == "verification_result":
            self._verification(data)
        elif event_type in {"plan_step_started", "plan_step_finished"}:
            self._plan_step(data)
        elif event_type == "auto_plan_started":
            source = self._safe(data.get("source", "controller"), limit=40)
            reason = self._safe(data.get("reason", "complex task"), limit=300)
            self._print_trace_label(
                "PLAN",
                f"entered read-only planning ({source}) · {reason}",
                style="bold cyan",
            )
        elif event_type == "tool_result":
            self._tool_result(data)

    def _reasoning(self, data: dict[str, Any]) -> None:
        if self.reasoning_mode == ReasoningDisplayMode.OFF:
            return
        phase = self._safe(data.get("phase", "decision")).upper()
        summary = self._safe(data.get("summary", ""), limit=500)
        if self.reasoning_mode == ReasoningDisplayMode.SUMMARY:
            self._print_trace_label(phase, summary, style="bold magenta")
            return

        lines = Text()
        lines.append(f"[{phase}]", style="bold magenta")
        lines.append(f"\nGoal: {self._safe(data.get('current_goal', ''), limit=300)}")
        lines.append(f"\nSummary: {summary}")
        assumptions = data.get("assumptions")
        if isinstance(assumptions, list) and assumptions:
            lines.append(f"\nAssumptions: {', '.join(self._safe(item) for item in assumptions)}")
        questions = data.get("open_questions")
        if isinstance(questions, list) and questions:
            lines.append(f"\nOpen questions: {', '.join(self._safe(item) for item in questions)}")
        next_action = data.get("next_action")
        if isinstance(next_action, dict):
            tool = self._safe(next_action.get("tool_name", ""))
            argument = self._safe(next_action.get("argument_summary", ""), limit=300)
            expected = self._safe(next_action.get("expected_result", ""), limit=300)
            lines.append(f"\nNext: {tool} {argument}".rstrip())
            lines.append(f"\nExpected: {expected}")
        confidence = data.get("confidence")
        if isinstance(confidence, (int, float)):
            lines.append(f"\nConfidence: {confidence:.2f}")
        self.console.print(lines)

    def _action(self, data: dict[str, Any]) -> None:
        if self.reasoning_mode == ReasoningDisplayMode.OFF:
            return
        tool = self._safe(data.get("tool", "unknown tool"))
        argument = self._safe(data.get("argument_summary", ""), limit=200)
        detail = f"{tool} {argument}".rstrip()
        self._print_trace_label("ACTION", detail, style="bold cyan")

    def _action_blocked(self, data: dict[str, Any]) -> None:
        if self.reasoning_mode == ReasoningDisplayMode.OFF:
            return
        tool = self._safe(data.get("tool", "unknown tool"))
        reason = self._safe(data.get("reason", "decision protocol rejected the call"), limit=300)
        self._print_trace_label("BLOCKED", self._join(tool, reason), style="bold yellow")

    def _observation(self, data: dict[str, Any]) -> None:
        if self.reasoning_mode == ReasoningDisplayMode.OFF:
            return
        summary = self._safe(data.get("result_summary", ""), limit=500)
        style = "bold green" if data.get("ok") is True else "bold red"
        self._print_trace_label("OBSERVE", summary, style=style)

    def _assessment(self, data: dict[str, Any]) -> None:
        if self.reasoning_mode == ReasoningDisplayMode.OFF:
            return
        summary = self._safe(data.get("summary", ""), limit=500)
        self._print_trace_label("ASSESS", summary, style="bold magenta")

    def _verification(self, data: dict[str, Any]) -> None:
        check_id = self._safe(data.get("check_id", "unknown"), limit=100)
        status = self._safe(data.get("status", "unknown"), limit=30)
        reasons = data.get("reasons")
        reason = None
        if isinstance(reasons, list) and reasons:
            reason = self._safe(reasons[0], limit=300)
        detail = self._join(f"{check_id} {status}", reason)
        style = "bold green" if status == "passed" else "bold red"
        self._print_trace_label("VERIFY", detail, style=style)

    def _plan_step(self, data: dict[str, Any]) -> None:
        index = self._safe(data.get("step_index", "?"), limit=10)
        total = self._safe(data.get("step_count", "?"), limit=10)
        title = self._safe(data.get("title", "plan step"), limit=160)
        status = self._safe(data.get("status", "unknown"), limit=30)
        error = self._safe(data.get("error", ""), limit=160) or None
        detail = self._join(title, status, error)
        style = "bold red" if status == "failed" else "bold cyan"
        if status == "blocked":
            style = "bold yellow"
        if status == "completed":
            style = "bold green"
        self._print_trace_label(f"PLAN {index}/{total}", detail, style=style)

    def _approval_decided(self, data: dict[str, Any]) -> None:
        action = self._safe(data.get("action", "unknown")).lower()
        allowed = action == "allow"
        source_value = self._safe(data.get("source", "unknown"))
        if allowed and source_value in _QUIET_ALLOW_SOURCES:
            return
        icon = "✓" if allowed else "✗"
        if source_value.startswith("semantic_template:"):
            template = source_value.partition(":")[2].replace("_", " ")
            source = f"semantic rule ({template})"
        else:
            source = _SOURCE_LABELS.get(source_value, source_value.replace("_", " "))
        status = "approved" if allowed else "denied"
        risk = self._safe(data.get("risk", "unknown")).lower()
        detail = self._join(
            f"{status} by {source}",
            f"risk {risk}",
            self._request_target(data) if not allowed else None,
            self._decision_reason(data.get("reason")) if not allowed else None,
            self._approval_guidance(data.get("guidance")) if not allowed else None,
        )
        self._print(
            icon,
            self._safe(data.get("tool", "unknown tool")),
            detail,
            style="bold green" if allowed else "bold red",
        )

    def _tool_started(self, data: dict[str, Any]) -> None:
        tool = self._safe(data.get("tool", "unknown tool"))
        progress = _TOOL_PROGRESS_LABELS.get(tool, "running tool")
        if self.console.is_terminal:
            sentence = f"{progress[:1].upper()}{progress[1:]} ({tool})…"
            self._start_status(f"[cyan]{sentence}[/cyan]")
            return
        self._print("…", tool, progress, style="bold cyan")

    def _start_model_status(self) -> None:
        self._start_status("[cyan]Model is generating the next action…[/cyan]")

    def _start_status(self, message: str) -> None:
        self._stop_status()
        self._active_status = self.console.status(message, spinner="dots")
        self._active_status.start()

    def _stop_status(self) -> None:
        if self._active_status is None:
            return
        self._active_status.stop()
        self._active_status = None

    def _tool_result(self, data: dict[str, Any]) -> None:
        if data.get("name") in {
            "record_decision",
            "register_verification",
            "run_verification",
        }:
            return
        if self.reasoning_mode != ReasoningDisplayMode.OFF:
            return
        ok = data.get("ok") is True
        if not ok and data.get("error_code") in _APPROVAL_FAILURE_CODES:
            return
        icon = "✓" if ok else "✗"
        metadata = data.get("metadata")
        exit_code = None
        duration = None
        if isinstance(metadata, dict):
            if isinstance(metadata.get("exit_code"), int):
                exit_code = f"exit {metadata['exit_code']}"
            if isinstance(metadata.get("duration_seconds"), (int, float)):
                duration = f"{metadata['duration_seconds']:g}s"
        error = None
        if not ok:
            error_value = data.get("error_code") or data.get("error")
            if error_value is not None:
                error = self._safe(error_value, limit=140)
        detail = self._join(
            None if ok else "failed",
            exit_code,
            duration,
            error,
        )
        self._print(
            icon,
            self._safe(data.get("name", "unknown tool")),
            detail,
            style="bold green" if ok else "bold red",
        )

    def _print(self, icon: str, label: str, detail: str, *, style: str) -> None:
        line = Text()
        line.append(f"{icon} {label}", style=style)
        if detail:
            line.append(f" · {detail}")
        self.console.print(line)

    def _print_trace_label(self, label: str, detail: str, *, style: str) -> None:
        line = Text()
        line.append(f"[{label}]", style=style)
        if detail:
            line.append(f" {detail}")
        self.console.print(line)

    def _safe(self, value: Any, *, limit: int = 180) -> str:
        text = str(value)
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        text = escape_terminal_controls(text)
        text = _INLINE_SECRET_PATTERN.sub(r"\1\2[REDACTED]", text)
        text = _INTERNAL_EVENT_ID_PATTERN.sub("", text)
        text = " ".join(text.split())
        if len(text) > limit:
            return f"{text[:limit]}…"
        return text

    @staticmethod
    def _join(*parts: str | None) -> str:
        return " · ".join(part for part in parts if part)

    def _decision_reason(self, value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        return self._safe(value, limit=120)

    def _approval_guidance(self, value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        return f"direction: {self._safe(value, limit=200)}"

    def _request_target(self, data: dict[str, Any]) -> str | None:
        program = data.get("program")
        if isinstance(program, str) and program:
            argument_count = data.get("argument_count")
            suffix = ""
            if isinstance(argument_count, int):
                noun = "arg" if argument_count == 1 else "args"
                suffix = f" ({argument_count} {noun})"
            return f"program {self._safe(program, limit=80)}{suffix}"
        affected_paths = data.get("affected_paths")
        if isinstance(affected_paths, list) and affected_paths:
            return f"target {self._safe(affected_paths[0], limit=100)}"
        return None
