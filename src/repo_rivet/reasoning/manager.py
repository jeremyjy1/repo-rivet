"""Create bounded reasoning and executor-owned observation records."""

import re
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from repo_rivet.memory.models import MemoryState, add_unique
from repo_rivet.reasoning.models import (
    ActionIntent,
    ObservationEvent,
    ReasoningConfig,
    ReasoningEvent,
    ReasoningPhase,
    RecordDecisionArgs,
)
from repo_rivet.tools.base import ToolCall, ToolResult

_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)


class ReasoningManager:
    """Validate, redact, compact, and query the durable decision trace."""

    def __init__(
        self,
        config: ReasoningConfig | None = None,
        *,
        secrets: tuple[str, ...] = (),
    ) -> None:
        self.config = config or ReasoningConfig()
        self._secrets = tuple(secret for secret in secrets if secret)

    def record(
        self,
        arguments: dict[str, Any],
        *,
        memory: MemoryState,
        step: int,
    ) -> ReasoningEvent:
        try:
            parsed = RecordDecisionArgs.model_validate(arguments)
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in error.errors()
            )
            raise ValueError(f"Invalid decision record: {details}") from None
        if len(parsed.summary) > self.config.max_summary_chars:
            raise ValueError(f"Decision summary exceeds {self.config.max_summary_chars} characters")
        next_action = None
        if parsed.next_tool is not None and parsed.expected_result is not None:
            next_action = ActionIntent(
                tool_name=parsed.next_tool,
                argument_summary=self._redact(parsed.next_tool_argument_summary or ""),
                expected_result=self._redact(parsed.expected_result),
            )
        event = ReasoningEvent(
            event_id=f"reason-{uuid4().hex[:12]}",
            session_id=memory.session_id,
            step=step,
            phase=parsed.phase,
            current_goal=self._redact(parsed.current_goal),
            summary=self._redact(parsed.summary),
            evidence_refs=[self._redact(item) for item in parsed.evidence_refs],
            assumptions=[self._redact(item) for item in parsed.assumptions],
            open_questions=[self._redact(item) for item in parsed.open_questions],
            next_action=next_action,
            confidence=parsed.confidence,
        )
        memory.reasoning_events.append(event)
        memory.reasoning_events[:] = memory.reasoning_events[-self.config.recent_event_limit :]
        self._update_structured_memory(memory, event)
        return event

    def observe(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        memory: MemoryState,
        step: int,
        output_ref: str | None,
    ) -> ObservationEvent:
        metadata = result.metadata or {}
        exit_code = metadata.get("exit_code")
        summary = self._redact(self._observation_summary(call, result))
        affected_paths = [self._redact(path) for path in self._affected_paths(call, metadata)]
        event = ObservationEvent(
            event_id=f"obs-{uuid4().hex[:12]}",
            session_id=memory.session_id,
            step=step,
            tool_call_id=call.id,
            tool_name=call.name,
            ok=result.ok,
            result_summary=summary,
            output_ref=output_ref,
            exit_code=exit_code if isinstance(exit_code, int) else None,
            affected_paths=affected_paths,
        )
        memory.observation_events.append(event)
        memory.observation_events[:] = memory.observation_events[-self.config.recent_event_limit :]
        return event

    def result_with_evidence(self, result: ToolResult, observation: ObservationEvent) -> ToolResult:
        metadata = dict(result.metadata or {})
        metadata["evidence_ref"] = observation.event_id
        return ToolResult(
            ok=result.ok,
            output=result.output,
            error=result.error,
            metadata=metadata,
            raw_output=result.raw_output,
            error_code=result.error_code,
            retryable=result.retryable,
        )

    def _update_structured_memory(self, memory: MemoryState, event: ReasoningEvent) -> None:
        if event.phase == ReasoningPhase.PLAN:
            memory.working.current_plan = [event.summary]
        elif event.phase == ReasoningPhase.DECISION:
            add_unique(memory.summary.key_decisions, event.summary)
        elif event.phase == ReasoningPhase.REFLECTION:
            add_unique(memory.summary.key_decisions, f"Reflection: {event.summary}")
        if event.next_action is not None:
            memory.working.pending_actions = [
                f"{event.next_action.tool_name}: {event.next_action.argument_summary}"
            ]
        elif event.phase == ReasoningPhase.FINAL_ASSESSMENT:
            memory.working.pending_actions.clear()

    def _redact(self, value: str) -> str:
        redacted = value
        for secret in self._secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        return _INLINE_SECRET_PATTERN.sub(r"\1\2[REDACTED]", redacted)

    @staticmethod
    def _affected_paths(call: ToolCall, metadata: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for value in (metadata.get("path"), call.arguments.get("path"), call.arguments.get("cwd")):
            if isinstance(value, str) and value and value not in paths:
                paths.append(value)
        return paths

    @staticmethod
    def _observation_summary(call: ToolCall, result: ToolResult) -> str:
        metadata = result.metadata or {}
        if not result.ok:
            code = result.error_code or "tool_error"
            detail = " ".join((result.error or "unknown error").split())[:700]
            return f"{call.name} failed ({code}): {detail}"
        if call.name == "list_files":
            return f"Listed {metadata.get('entries', 'unknown')} workspace entries."
        if call.name == "search_text":
            count = metadata.get("matches", "unknown")
            locations = metadata.get("match_locations")
            if isinstance(locations, list) and locations:
                suffix = ", ".join(str(location) for location in locations)
                if isinstance(count, int) and count > len(locations):
                    suffix = f"{suffix}, …"
                return f"Found {count} matching lines at {suffix}."
            return f"Found {count} matching lines."
        if call.name == "read_file":
            path = metadata.get("path") or call.arguments.get("path", "file")
            if metadata.get("total_lines") == 0:
                return f"Read empty file {path}."
            if (
                "fully_visible_end_line" in metadata
                and metadata.get("fully_visible_end_line") is None
            ):
                return (
                    f"Partially read {path}:{metadata.get('start_line', '?')}; "
                    "the truncated line is not editable."
                )
            start = metadata.get("start_line", "?")
            end = metadata.get("fully_visible_end_line", metadata.get("end_line", "?"))
            return f"Read {path}:{start}-{end}."
        if call.name == "write_file":
            path = metadata.get("path") or call.arguments.get("path", "file")
            line_count = metadata.get("line_count")
            location = (
                f"{path}:1-{line_count}" if isinstance(line_count, int) and line_count else path
            )
            return f"Wrote {metadata.get('bytes', 'unknown')} bytes to {location}."
        if call.name == "edit_file":
            path = metadata.get("path") or call.arguments.get("path", "file")
            changed_ranges = metadata.get("changed_ranges")
            if isinstance(changed_ranges, list) and changed_ranges:
                rendered = ", ".join(
                    f"{path}:{item[0]}-{item[1]}"
                    for item in changed_ranges
                    if isinstance(item, list) and len(item) == 2
                )
                return f"Committed snapshot-anchored edits at {rendered}."
            return f"Committed snapshot-anchored edits in {path}."
        if call.name in {"run_command", "run_verification"}:
            return f"Command finished with exit code {metadata.get('exit_code', 'unknown')}."
        if call.name == "git_diff":
            return "Inspected the current Git diff."
        return f"{call.name} completed successfully."
