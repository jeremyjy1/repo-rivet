"""Thread-safe human approval bridge for the browser client."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from repo_rivet.approval.models import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalScope,
    LLMReviewResult,
)


class EventSink(Protocol):
    def log(self, event_type: str, **data: Any) -> None: ...


@dataclass(slots=True)
class PendingApproval:
    request: ApprovalRequest
    llm_review: LLMReviewResult | None
    state_version: int
    ready: threading.Event
    decision: ApprovalDecision | None = None


class WebHumanApprover:
    def __init__(self, *, event_logger: EventSink | None = None) -> None:
        self._condition = threading.Condition()
        self._pending: PendingApproval | None = None
        self._version = 0
        self._event_logger = event_logger

    def bind_event_logger(self, event_logger: EventSink | None) -> None:
        """Attach the run's durable event sink before approvals can be requested."""
        with self._condition:
            self._event_logger = event_logger

    def ask(
        self,
        request: ApprovalRequest,
        *,
        llm_review: LLMReviewResult | None = None,
    ) -> ApprovalDecision:
        with self._condition:
            self._version += 1
            pending = PendingApproval(request, llm_review, self._version, threading.Event())
            self._pending = pending
            self._condition.notify_all()
        self._log(
            "approval_awaiting_human",
            request_id=request.request_id,
            tool=request.tool_name,
            risk=request.assessment.level.name.lower(),
            state_version=pending.state_version,
            llm_review_available=llm_review is not None,
        )
        pending.ready.wait()
        with self._condition:
            decision = pending.decision
            if self._pending is pending:
                self._pending = None
        if decision is None:
            return self._decision(request, "stop", "Run stopped while approval was pending")
        return decision

    def wait_for_pending(self, timeout: float = 0) -> PendingApproval | None:
        with self._condition:
            if self._pending is None and timeout > 0:
                self._condition.wait(timeout)
            return self._pending

    def resolve(
        self,
        *,
        request_id: str,
        state_version: int,
        action: Literal["allow_once", "allow_session", "deny", "stop"],
        guidance: str | None = None,
    ) -> ApprovalDecision:
        with self._condition:
            pending = self._pending
            if pending is None:
                raise ValueError("No approval is pending")
            if pending.request.request_id != request_id or pending.state_version != state_version:
                raise ValueError("Approval request is stale")
            if pending.decision is not None:
                raise ValueError("Approval request was already resolved")
            if action == "allow_session" and pending.request.assessment.level.name in {
                "HIGH",
                "CRITICAL",
            }:
                raise ValueError("High-risk requests cannot receive a repeating session grant")
            decision = self._decision(pending.request, action, guidance)
            pending.decision = decision
            pending.ready.set()
            self._condition.notify_all()
            return decision

    def abort_pending(self) -> None:
        with self._condition:
            pending = self._pending
            if pending is not None and pending.decision is None:
                pending.decision = self._decision(pending.request, "stop", None)
                pending.ready.set()

    def snapshot(self) -> dict[str, object] | None:
        with self._condition:
            pending = self._pending
            if pending is None or pending.decision is not None:
                return None
            request = pending.request
            facts = request.facts
            review = pending.llm_review
            presentation = _approval_presentation(request)
            return {
                "request_id": request.request_id,
                "state_version": pending.state_version,
                "tool": request.tool_name,
                "risk": request.assessment.level.name.lower(),
                "reasons": request.assessment.reasons,
                "operation": facts.operation_class.value,
                "analysis": facts.analysis_level.value,
                "reads": [_relative_path(path, request.workspace) for path in facts.read_paths],
                "writes": [_relative_path(path, request.workspace) for path in facts.write_paths],
                "deletes": [_relative_path(path, request.workspace) for path in facts.delete_paths],
                "effects": sorted(facts.explicit_effects),
                **presentation,
                "review": (
                    {
                        "recommendation": review.recommendation,
                        "risk": review.risk_level,
                        "relevance": review.task_relevance,
                        "reason": review.reason,
                        "question": review.user_prompt,
                        "unknowns": review.unknowns,
                        "constraints": review.required_constraints,
                    }
                    if review
                    else None
                ),
                "allow_matching_repeats": request.assessment.level.name not in {"HIGH", "CRITICAL"},
            }

    def _log(self, event_type: str, **data: Any) -> None:
        with self._condition:
            event_logger = self._event_logger
        if event_logger is not None:
            event_logger.log(event_type, **data)

    @staticmethod
    def _decision(
        request: ApprovalRequest,
        action: str,
        guidance: str | None,
    ) -> ApprovalDecision:
        if action in {"allow_once", "allow_session"}:
            return ApprovalDecision(
                action=ApprovalAction.ALLOW,
                source="web_human",
                reason="approved by user",
                risk_level=request.assessment.level,
                request_fingerprint=request.fingerprint,
                scope=(
                    ApprovalScope.SESSION_EXACT if action == "allow_session" else ApprovalScope.ONCE
                ),
            )
        normalized_guidance = " ".join((guidance or "").split())[:1000] or None
        abort = action == "stop"
        return ApprovalDecision(
            action=ApprovalAction.DENY,
            source="web_human",
            reason="run stopped by user" if abort else "denied by user",
            risk_level=request.assessment.level,
            request_fingerprint=request.fingerprint,
            abort_agent=abort,
            guidance=None if abort else normalized_guidance,
        )


_DETAIL_LABELS = {
    "case_sensitive": "Case sensitive",
    "check_id": "Verification check",
    "cwd": "Working directory",
    "end_line": "End line",
    "max_depth": "Maximum depth",
    "path": "Target",
    "query": "Search query",
    "regex": "Regular expression",
    "start_line": "Start line",
    "timeout_seconds": "Timeout",
}
_HIDDEN_ARGUMENTS = {
    "_outside_workspace_paths",
    "_resolved_paths",
    "content",
    "diff_preview",
    "fingerprint",
    "operations",
    "prepared_live_hash",
    "snapshot_id",
    "snapshot_tag",
    "stdin",
}


def _approval_presentation(request: ApprovalRequest) -> dict[str, object]:
    """Convert normalized tool arguments into bounded, human-facing display fields."""
    arguments = request.normalized_arguments
    facts = request.facts
    details: list[dict[str, str]] = []
    if request.tool_name in {"run_command", "run_verification"}:
        details.extend(
            [
                {"label": "Analysis", "value": facts.analysis_level.value},
                {
                    "label": "Executable origin",
                    "value": facts.executable_origin.value.replace("_", " "),
                },
                {
                    "label": "Effect scope",
                    "value": facts.effect_scope.value.replace("_", " "),
                },
            ]
        )
        if facts.resolved_executable:
            details.append({"label": "Resolved executable", "value": facts.resolved_executable})
        if facts.output_provenance:
            ownership = ", ".join(
                f"{_relative_path(path, request.workspace)}: {value.value.replace('_', ' ')}"
                for path, value in facts.output_provenance.items()
            )
            details.append({"label": "Output ownership", "value": ownership})
        if facts.potential_capabilities:
            details.append(
                {
                    "label": "Possible effects",
                    "value": ", ".join(sorted(facts.potential_capabilities)),
                }
            )

    command = arguments.get("command")
    if isinstance(command, dict):
        details.append({"label": "Program", "value": _display_value(command.get("program"))})
        command_arguments = command.get("args")
        if isinstance(command_arguments, list):
            details.append(
                {
                    "label": "Arguments",
                    "value": " ".join(_display_value(item) for item in command_arguments)
                    or "(none)",
                }
            )

    for key, value in arguments.items():
        if key in _HIDDEN_ARGUMENTS or key == "command" or key.startswith("_") or value is None:
            continue
        rendered = _display_value(value)
        if key == "timeout_seconds":
            rendered = f"{rendered} seconds"
        details.append(
            {
                "label": _DETAIL_LABELS.get(key, key.replace("_", " ").capitalize()),
                "value": rendered,
            }
        )

    operations = arguments.get("operations")
    operation_descriptions = (
        [_operation_description(item) for item in operations]
        if isinstance(operations, list)
        else []
    )
    preview: dict[str, str] | None = None
    diff = arguments.get("diff_preview")
    if isinstance(diff, str):
        preview = {"kind": "diff", "title": "Proposed changes", "text": diff}
    elif request.tool_name == "write_file":
        content = request.arguments.get("content")
        if isinstance(content, str):
            preview = {
                "kind": "content",
                "title": "New file content",
                "text": _bounded_preview(content),
            }
    return {
        "details": details,
        "operations": operation_descriptions,
        "preview": preview,
    }


def _relative_path(path: str, workspace: str) -> str:
    candidate = Path(path)
    try:
        return candidate.relative_to(Path(workspace)).as_posix() or "."
    except ValueError:
        return str(candidate)


def _display_value(value: object) -> str:
    if isinstance(value, dict):
        if value.get("redacted") is True:
            return "[REDACTED]"
        characters = value.get("characters")
        if isinstance(characters, int):
            return f"{characters} characters"
        return "(details hidden)"
    if isinstance(value, list):
        return ", ".join(_display_value(item) for item in value) or "(none)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value if value is not None else "unknown")


def _operation_description(value: object) -> str:
    if not isinstance(value, dict):
        return "Unrecognized edit operation"
    operation = str(value.get("op", "unknown"))
    count = value.get("new_line_count")
    line_count = count if isinstance(count, int) else 0
    line_word = "line" if line_count == 1 else "lines"
    if operation == "replace":
        return (
            f"Replace lines {value.get('start_line', '?')}-{value.get('end_line', '?')} "
            f"with {line_count} {line_word}"
        )
    if operation == "delete":
        return f"Delete lines {value.get('start_line', '?')}-{value.get('end_line', '?')}"
    if operation in {"insert_before", "insert_after"}:
        position = "before" if operation == "insert_before" else "after"
        return f"Insert {line_count} {line_word} {position} line {value.get('line', '?')}"
    if operation == "insert_start":
        return f"Insert {line_count} {line_word} at the start of the file"
    if operation == "insert_end":
        return f"Insert {line_count} {line_word} at the end of the file"
    return f"Unrecognized edit operation: {operation}"


def _bounded_preview(content: str, limit: int = 20_000) -> str:
    if len(content) <= limit:
        return content
    return f"{content[:limit]}\n\n… {len(content) - limit:,} additional characters omitted"
