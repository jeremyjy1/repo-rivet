"""Interactive and fail-closed non-interactive approval adapters."""

import json
import select
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from rich.console import Console, ConsoleRenderable, Group, RichCast
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from repo_rivet.approval.models import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
    LLMReviewResult,
    NonInteractivePolicy,
)
from repo_rivet.storage.terminal_text import escape_terminal_controls

_HIDDEN_DISPLAY_KEYS = frozenset(
    {
        "fingerprint",
        "prepared_live_hash",
        "snapshot_id",
        "snapshot_tag",
    }
)
_DISPLAY_LABELS = {
    "case_sensitive": "Case sensitive",
    "check_id": "Verification check",
    "content": "Content",
    "cwd": "Working directory",
    "end_line": "End line",
    "max_depth": "Maximum depth",
    "path": "File",
    "query": "Search query",
    "regex": "Regular expression",
    "start_line": "Start line",
    "timeout_seconds": "Timeout",
}
_Renderable = ConsoleRenderable | RichCast | str


class HumanApprover(Protocol):
    def ask(
        self,
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None = None,
    ) -> ApprovalDecision:
        """Obtain or synthesize a human approval decision."""
        ...


class TerminalHumanApprover:
    """Show a bounded request summary and accept an explicit terminal choice."""

    def __init__(
        self,
        console: Console,
        *,
        reader: Callable[[str], str] | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.console = console
        self.reader = reader
        self.timeout_seconds = timeout_seconds

    def ask(
        self,
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None = None,
    ) -> ApprovalDecision:
        self.console.print(self._build_panel(request, llm_review=llm_review))
        prompt = "Select approval option [1-4]"
        while True:
            try:
                choice = self._read_choice(prompt)
            except (EOFError, KeyboardInterrupt):
                choice = "4"
            if choice is None:
                return ApprovalDecision(
                    action=ApprovalAction.DENY,
                    source="human_timeout",
                    reason="approval timed out without a user decision",
                    risk_level=request.assessment.level,
                    request_fingerprint=request.fingerprint,
                )
            choice = choice.strip().lower()
            decision = self._decision_for_choice(request, choice)
            if decision is not None:
                if decision.action == ApprovalAction.DENY and not decision.abort_agent:
                    guidance = self._read_denial_reason()
                    if guidance is not None:
                        decision = decision.model_copy(update={"guidance": guidance})
                return decision
            self.console.print("Enter a number from 1 to 4.", style="yellow")

    def _read_denial_reason(self) -> str | None:
        prompt = "Reason for denial or direction for the agent (optional; press Enter to skip)"
        try:
            value = self._read_choice(prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        if value is None:
            return None
        normalized = " ".join("".join(character for character in value if character >= " ").split())
        return normalized[:1_000] or None

    @staticmethod
    def _build_panel(
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None,
    ) -> Panel:
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Tool", request.tool_name)
        summary.add_row("Risk", request.assessment.level.name)

        reasons = Text()
        for index, reason in enumerate(request.assessment.reasons):
            if index:
                reasons.append("\n")
            reasons.append(f"• {reason}")
        if not reasons:
            reasons.append("• No deterministic reason was provided.")

        sections: list[_Renderable] = [summary]
        if request.tool_name == "edit_file":
            sections.extend(TerminalHumanApprover._edit_sections(request))
        else:
            sections.extend(TerminalHumanApprover._request_sections(request))
        sections.extend((Text("\nRisk reasons", style="bold"), reasons))
        if llm_review is not None:
            review = Text()
            review.append(
                f"{llm_review.recommendation.upper()} · "
                f"risk {llm_review.risk_level} · "
                f"relevance {llm_review.task_relevance}\n",
                style="bold",
            )
            review.append(_display_value(llm_review.reason))
            if llm_review.recognized_effects:
                review.append("\nEffects: ")
                review.append(_display_value(llm_review.recognized_effects))
            if llm_review.unknowns:
                review.append("\nUnknowns: ", style="yellow")
                review.append("; ".join(_display_value(item) for item in llm_review.unknowns))
            if llm_review.required_constraints:
                review.append("\nConstraints: ")
                review.append(_display_value(llm_review.required_constraints))
            if llm_review.user_prompt:
                review.append("\nApproval question: ", style="bold yellow")
                review.append(_display_value(llm_review.user_prompt))
            sections.extend((Text("\nLLM review", style="bold"), review))

        options = Table.grid(padding=(0, 2))
        options.add_column(style="bold cyan", justify="right")
        options.add_column()
        options.add_row("1", "Allow once")
        options.add_row("2", "Allow matching repeats for this session")
        options.add_row("3", "Deny and continue")
        options.add_row("4", "Stop current run and save session")
        sections.extend((Text("\nOptions", style="bold"), options))
        title = (
            "Edit Approval Required" if request.tool_name == "edit_file" else "Approval Required"
        )
        return Panel(Group(*sections), title=title, border_style="yellow")

    @staticmethod
    def _edit_sections(request: ApprovalRequest) -> list[_Renderable]:
        arguments = request.normalized_arguments
        details = Table.grid(padding=(0, 2))
        details.add_column(style="bold")
        details.add_column()
        details.add_row("File", _display_value(arguments.get("path", "unknown")))
        operations = arguments.get("operations")
        operation_values = operations if isinstance(operations, list) else []
        details.add_row("Operations", str(len(operation_values)))

        operation_text = Text()
        for index, operation in enumerate(operation_values, start=1):
            if index > 1:
                operation_text.append("\n")
            operation_text.append(f"{index}. ", style="bold cyan")
            operation_text.append(_operation_description(operation))
        if not operation_values:
            operation_text.append("No edit operations were provided.", style="yellow")

        diff_value = arguments.get("diff_preview")
        return [
            Text("\nRequested edit", style="bold"),
            details,
            Text("\nOperations", style="bold"),
            operation_text,
            Text("\nProposed changes", style="bold"),
            _format_diff(diff_value if isinstance(diff_value, str) else ""),
        ]

    @staticmethod
    def _request_sections(request: ApprovalRequest) -> list[_Renderable]:
        rows = _request_rows(request)
        if not rows:
            return []
        details = Table.grid(padding=(0, 2))
        details.add_column(style="bold")
        details.add_column()
        for label, value in rows:
            details.add_row(label, value)
        return [Text("\nRequested action", style="bold"), details]

    def _read_choice(self, prompt: str) -> str | None:
        if self.reader is not None:
            return self.reader(prompt)
        if self.timeout_seconds is None or not sys.stdin.isatty():
            return Prompt.ask(prompt, console=self.console)
        self.console.print(f"{prompt}: ", end="")
        try:
            readable, _, _ = select.select([sys.stdin], [], [], self.timeout_seconds)
        except (OSError, ValueError):
            return Prompt.ask(prompt, console=self.console)
        if not readable:
            self.console.print()
            return None
        return sys.stdin.readline()

    @staticmethod
    def _decision_for_choice(
        request: ApprovalRequest,
        choice: str,
    ) -> ApprovalDecision | None:
        choices = {
            "1": (ApprovalAction.ALLOW, ApprovalScope.ONCE, False, "allowed once by user"),
            "2": (
                ApprovalAction.ALLOW,
                ApprovalScope.SESSION_EXACT,
                False,
                "allowed matching requests for this session",
            ),
            "3": (
                ApprovalAction.DENY,
                ApprovalScope.ONCE,
                False,
                "request denied by user; continue current run",
            ),
            "4": (
                ApprovalAction.DENY,
                ApprovalScope.ONCE,
                True,
                "current run stopped by user; session will be saved",
            ),
        }
        selected = choices.get(choice)
        if selected is None:
            return None
        action, scope, abort_agent, reason = selected
        return ApprovalDecision(
            action=action,
            source="human",
            reason=reason,
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
            scope=scope,
            abort_agent=abort_agent,
        )


def _request_rows(request: ApprovalRequest) -> list[tuple[str, str]]:
    arguments = _without_hashes(request.normalized_arguments)
    if not isinstance(arguments, dict):
        return []
    rows: list[tuple[str, str]] = []
    facts = request.facts
    if request.tool_name in {"run_command", "run_verification"}:
        rows.extend(
            [
                ("Operation", facts.operation_class.value.replace("_", " ")),
                ("Analysis", facts.analysis_level.value),
                ("Executable origin", facts.executable_origin.value.replace("_", " ")),
                ("Effect scope", facts.effect_scope.value.replace("_", " ")),
            ]
        )
        if facts.resolved_executable:
            rows.append(("Resolved executable", _display_value(facts.resolved_executable)))
        if facts.read_paths:
            rows.append(("Reads", _path_list(facts.read_paths, request.workspace)))
        if facts.write_paths:
            rows.append(("Writes", _path_list(facts.write_paths, request.workspace)))
            provenance = ", ".join(
                f"{_relative_path(path, request.workspace)}: {value.value.replace('_', ' ')}"
                for path, value in facts.output_provenance.items()
            )
            rows.append(("Output ownership", provenance))
        if facts.explicit_effects:
            rows.append(("Known effects", ", ".join(sorted(facts.explicit_effects))))
        if facts.potential_capabilities:
            rows.append(("Possible effects", ", ".join(sorted(facts.potential_capabilities))))
    path = arguments.get("path")
    if isinstance(path, str):
        path_label = (
            "Directory"
            if request.tool_name in {"git_diff", "git_status", "list_files", "search_text"}
            else "Path"
            if request.tool_name == "delete_path"
            else "File"
        )
        rows.append((path_label, _display_value(path)))

    command = arguments.get("command")
    if isinstance(command, dict):
        rows.append(("Program", _display_value(command.get("program", "unknown"))))
        command_arguments = command.get("args")
        if isinstance(command_arguments, list):
            rows.append(
                (
                    "Arguments",
                    " ".join(_display_value(value) for value in command_arguments) or "(none)",
                )
            )

    skipped = {
        "command",
        "diff_preview",
        "operations",
        "path",
    }
    for key, value in arguments.items():
        if key in skipped or key.startswith("_") or value is None:
            continue
        label = _DISPLAY_LABELS.get(key, key.replace("_", " ").capitalize())
        rendered = _display_value(value)
        if key == "timeout_seconds" and rendered not in {"", "unknown"}:
            rendered = f"{rendered} seconds"
        rows.append((label, rendered))
    return rows


def _path_list(paths: list[str], workspace: str) -> str:
    return ", ".join(_relative_path(path, workspace) for path in paths)


def _relative_path(path: str, workspace: str) -> str:
    candidate = Path(path)
    try:
        return candidate.relative_to(Path(workspace)).as_posix() or "."
    except ValueError:
        return str(candidate)


def _without_hashes(value: object) -> object:
    if isinstance(value, dict):
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if (
                normalized in _HIDDEN_DISPLAY_KEYS
                or normalized == "hash"
                or normalized == "sha256"
                or normalized.endswith("_sha256")
                or normalized.endswith("_hash")
                or normalized.endswith("_fingerprint")
                or normalized.endswith("snapshot_id")
                or normalized.endswith("snapshot_tag")
            ):
                continue
            cleaned[str(key)] = _without_hashes(item)
        return cleaned
    if isinstance(value, list):
        return [_without_hashes(item) for item in value]
    return value


def _display_value(value: object) -> str:
    if isinstance(value, dict):
        if value.get("redacted") is True:
            return "[REDACTED]"
        if set(value) == {"characters"} and isinstance(value.get("characters"), int):
            count = value["characters"]
            return f"{count} character" if count == 1 else f"{count} characters"
        value = _without_hashes(value)
        return escape_terminal_controls(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    if isinstance(value, list):
        return ", ".join(_display_value(item) for item in value) or "(none)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return escape_terminal_controls(str(value))


def _operation_description(value: object) -> str:
    if not isinstance(value, dict):
        return "Unrecognized edit operation"
    operation = str(value.get("op", "unknown"))
    count = value.get("new_line_count")
    line_count = count if isinstance(count, int) else 0
    new_lines = "line" if line_count == 1 else "lines"
    if operation == "replace":
        start = value.get("start_line", "?")
        end = value.get("end_line", "?")
        return f"Replace lines {start}-{end} with {line_count} {new_lines}"
    if operation == "delete":
        return f"Delete lines {value.get('start_line', '?')}-{value.get('end_line', '?')}"
    if operation == "insert_before":
        return f"Insert {line_count} {new_lines} before line {value.get('line', '?')}"
    if operation == "insert_after":
        return f"Insert {line_count} {new_lines} after line {value.get('line', '?')}"
    if operation == "insert_start":
        return f"Insert {line_count} {new_lines} at the start of the file"
    if operation == "insert_end":
        return f"Insert {line_count} {new_lines} at the end of the file"
    return f"Unrecognized edit operation: {escape_terminal_controls(operation)}"


def _format_diff(diff: str) -> Text:
    if not diff:
        return Text("(no textual changes)", style="dim")
    rendered = Text()
    for line in diff.splitlines(keepends=True):
        safe_line = escape_terminal_controls(line)
        if line.startswith("@@"):
            style = "bold cyan"
        elif line.startswith("+++") or line.startswith("---"):
            style = "bold"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        else:
            style = None
        rendered.append(safe_line, style=style)
    return rendered


class NonInteractiveHumanApprover:
    """Never infer approval when no human input channel exists."""

    def __init__(self, policy: NonInteractivePolicy = NonInteractivePolicy.DENY) -> None:
        self.policy = policy

    def ask(
        self,
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None = None,
    ) -> ApprovalDecision:
        if self.policy == NonInteractivePolicy.FAIL:
            reason = "human approval is required but unavailable in non-interactive mode"
        else:
            reason = "request denied because non-interactive mode cannot obtain approval"
        return ApprovalDecision(
            action=ApprovalAction.DENY,
            source="non_interactive_policy",
            reason=reason,
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
            abort_agent=self.policy == NonInteractivePolicy.FAIL,
        )
