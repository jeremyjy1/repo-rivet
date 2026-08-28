"""Typed fixed, working, summary, file, and command memory models."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repo_rivet.approval.models import ApprovalMode
from repo_rivet.memory.token_estimator import ApproximateTokenEstimator
from repo_rivet.reasoning.models import ObservationEvent, ReasoningEvent
from repo_rivet.tools.base import ToolCall, ToolResult


class MemoryConfig(BaseModel):
    """Context budget and compaction thresholds for one session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recent_message_limit: int = Field(default=10, ge=4, le=30)
    max_context_tokens: int = Field(default=24_000, ge=1_000)
    reserved_output_tokens: int = Field(default=4_000, ge=100)
    reserved_tool_result_tokens: int = Field(default=2_048, ge=0)
    safety_margin_ratio: float = Field(default=0.15, ge=0, lt=0.5)
    max_tool_output_chars: int = Field(default=12_000, ge=1_000)
    command_head_lines: int = Field(default=80, ge=1)
    command_tail_lines: int = Field(default=120, ge=1)
    max_file_read_lines: int = Field(default=300, ge=1)
    compaction_threshold: float = Field(default=0.70, gt=0, lt=1)
    hard_limit_threshold: float = Field(default=0.85, gt=0, le=1)
    default_correction_factor: float = Field(default=1.25, ge=1.0, le=3.0)
    calibration_window: int = Field(default=20, ge=1, le=100)
    max_context_overflow_retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def validate_budget(self) -> "MemoryConfig":
        safety_margin = int(self.max_context_tokens * self.safety_margin_ratio)
        total_reserve = (
            self.reserved_output_tokens + self.reserved_tool_result_tokens + safety_margin
        )
        if total_reserve >= self.max_context_tokens:
            raise ValueError(
                "output, tool-result, and safety reserves must leave a positive prompt budget"
            )
        if self.hard_limit_threshold <= self.compaction_threshold:
            raise ValueError("hard_limit_threshold must exceed compaction_threshold")
        return self


class Message(BaseModel):
    """A recent conversation message with native tool-call linkage."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    step: int = 0

    @classmethod
    def from_chat_message(cls, message: dict[str, Any], *, step: int = 0) -> "Message":
        return cls(
            role=message["role"],
            content=message.get("content"),
            tool_call_id=message.get("tool_call_id"),
            name=message.get("name"),
            tool_calls=message.get("tool_calls"),
            step=step,
        )

    def as_chat_message(self) -> dict[str, Any]:
        content = self._model_visible_content()
        message: dict[str, Any] = {"role": self.role, "content": content}
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            message["name"] = self.name
        if self.tool_calls is not None:
            message["tool_calls"] = self.tool_calls
        return message

    def _model_visible_content(self) -> str | None:
        """Hide session-audit paths that workspace tools cannot read."""
        if self.role != "tool" or not self.content:
            return self.content
        try:
            payload = json.loads(self.content)
        except (TypeError, json.JSONDecodeError):
            return self.content
        if not isinstance(payload, dict):
            return self.content
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("output_ref", None)
        output = payload.get("output")
        if isinstance(output, str):
            payload["output"] = "\n".join(
                line for line in output.splitlines() if not line.startswith("Full output: ")
            )
        return json.dumps(payload, ensure_ascii=False)


class FixedMemory(BaseModel):
    """Session facts that compaction must never remove or rewrite."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    original_task: str
    workspace: str
    safety_rules: list[str]
    completion_rules: list[str]
    max_steps: int


class WorkingMemory(BaseModel):
    """High-value mutable facts for the next model decision."""

    current_plan: list[str] = Field(default_factory=list)
    current_focus: str | None = None
    unresolved_errors: list[str] = Field(default_factory=list)
    pending_actions: list[str] = Field(default_factory=list)
    last_error: str | None = None
    last_verification_result: str | None = None
    recent_modified_files: list[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    """Structured long-term summary updated from deterministic events."""

    task_goal: str = ""
    completed_actions: list[str] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    verification_status: str = "not run"
    unresolved_issues: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)

    def has_content(self) -> bool:
        return (
            any(
                (
                    self.completed_actions,
                    self.key_decisions,
                    self.files_read,
                    self.files_modified,
                    self.commands_run,
                    self.unresolved_issues,
                    self.next_actions,
                )
            )
            or self.verification_status != "not run"
        )


class FileMemory(BaseModel):
    """The latest known version of one workspace file."""

    path: str
    sha256: str
    last_read_step: int
    last_modified_step: int | None = None
    content_preview: str | None = None


class CommandOutputMemory(BaseModel):
    """Bounded command observation plus a reference to full local output."""

    command: str
    exit_code: int | None
    head: str
    tail: str
    context_output: str = ""
    original_chars: int = 0
    estimated_tokens: int = 0
    full_output_path: str | None = None
    truncated: bool = False


class MemoryState(BaseModel):
    """Serializable state for one RepoRivet session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    config: MemoryConfig = Field(default_factory=MemoryConfig)
    fixed: FixedMemory | None = None
    task_updates: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    summary: ConversationSummary = Field(default_factory=ConversationSummary)
    working: WorkingMemory = Field(default_factory=WorkingMemory)
    file_memories: dict[str, FileMemory] = Field(default_factory=dict)
    invalidated_files: set[str] = Field(default_factory=set)
    modified_files: set[str] = Field(default_factory=set)
    command_outputs: list[CommandOutputMemory] = Field(default_factory=list)
    last_file_change_step: int | None = None
    last_verification_step: int | None = None
    last_verification_success: bool = False
    tool_event_step: int = 0
    compaction_count: int = 0
    context_overflow_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    approval_session_grants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    denied_request_fingerprints: set[str] = Field(default_factory=set)
    approval_denial_guidance: dict[str, str] = Field(default_factory=dict)
    approval_mode_override: ApprovalMode | None = None
    reasoning_events: list[ReasoningEvent] = Field(default_factory=list)
    observation_events: list[ObservationEvent] = Field(default_factory=list)
    reflection_required: bool = False
    status: str = "ready"

    def start_task(
        self,
        *,
        task: str,
        workspace: str,
        system_prompt: str,
        safety_rules: list[str],
        completion_rules: list[str],
        max_steps: int,
    ) -> None:
        """Preserve the first task verbatim and version every later user request."""
        normalized = task.strip()
        self.denied_request_fingerprints.clear()
        self.approval_denial_guidance.clear()
        if self.fixed is None:
            self.fixed = FixedMemory(
                system_prompt=system_prompt,
                original_task=normalized,
                workspace=workspace,
                safety_rules=safety_rules,
                completion_rules=completion_rules,
                max_steps=max_steps,
            )
            self.summary.task_goal = normalized
        elif normalized != self.fixed.original_task and normalized not in self.task_updates:
            self.task_updates.append(normalized)

        self.messages.append(Message(role="user", content=normalized, step=self.tool_event_step))
        self.working.current_focus = normalized
        self.working.pending_actions.clear()
        self.status = "running"

    def append_assistant(self, message: dict[str, Any], *, step: int) -> None:
        self.messages.append(Message.from_chat_message(message, step=step))

    def append_tool_message(self, message: dict[str, Any], *, step: int) -> None:
        self.messages.append(Message.from_chat_message(message, step=step))

    def record_tool_result(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        step: int,
        full_output_path: str | None = None,
    ) -> None:
        """Update structured memory and append a bounded, linked tool observation."""
        from repo_rivet.agent.verifier import is_verification_command

        self.tool_event_step = max(self.tool_event_step, step)
        metadata = dict(result.metadata or {})
        context_output = result.output
        path = call.arguments.get("path")

        if call.name == "read_file" and result.ok and isinstance(path, str):
            sha256 = metadata.get("sha256")
            if isinstance(sha256, str):
                existing = self.file_memories.get(path)
                if existing and existing.sha256 == sha256:
                    existing.last_read_step = step
                    existing.content_preview = (result.raw_output or result.output)[:2_000]
                else:
                    was_invalidated = path in self.invalidated_files
                    self.file_memories[path] = FileMemory(
                        path=path,
                        sha256=sha256,
                        last_read_step=step,
                        content_preview=(result.raw_output or result.output)[:2_000],
                    )
                    self.invalidated_files.discard(path)
                    if was_invalidated:
                        context_output = (
                            f"Latest version after prior invalidation:\n{context_output}"
                        )
                add_unique(self.summary.files_read, path)

        elif call.name in {"write_file", "replace_text"} and result.ok and isinstance(path, str):
            self.file_memories.pop(path, None)
            self.invalidated_files.add(path)
            self.modified_files.add(path)
            self.last_file_change_step = step
            self.last_verification_success = False
            add_unique(self.summary.files_modified, path)
            add_unique(self.summary.completed_actions, f"Modified {path} with {call.name}")
            add_unique(self.working.recent_modified_files, path, limit=20)
            context_output = (
                f"{context_output}\nPrevious read memory for {path} is now invalid. "
                "Read the file again before relying on its content."
            )

        elif call.name == "run_command":
            command = call.arguments.get("command")
            if isinstance(command, str):
                exit_code = metadata.get("exit_code")
                raw_lines = (result.raw_output or result.output).splitlines()
                head = "\n".join(raw_lines[: self.config.command_head_lines])
                tail = "\n".join(raw_lines[-self.config.command_tail_lines :])
                command_memory = CommandOutputMemory(
                    command=command,
                    exit_code=exit_code if isinstance(exit_code, int) else None,
                    head=head,
                    tail=tail,
                    original_chars=len(result.raw_output or result.output),
                    full_output_path=full_output_path,
                    truncated=bool(metadata.get("truncated")),
                )
                self.command_outputs.append(command_memory)
                self.command_outputs[:] = self.command_outputs[-20:]
                add_unique(self.summary.commands_run, f"{command} (exit {exit_code})")

                if exit_code == 0 and len(raw_lines) > 20:
                    context_output = (
                        f"Command succeeded: {command}\nSummary tail:\n{'\n'.join(raw_lines[-10:])}"
                    )
                if full_output_path:
                    context_output = (
                        f"{context_output}\nFull output is retained in session audit storage "
                        "outside the workspace and cannot be read with workspace tools."
                    )
                command_memory.context_output = context_output
                command_memory.estimated_tokens = ApproximateTokenEstimator().estimate_text(
                    context_output,
                    kind="log",
                )

                if is_verification_command(command):
                    success = result.ok and exit_code == 0 and not metadata.get("timed_out")
                    self.last_verification_step = step
                    self.last_verification_success = success
                    self.summary.verification_status = (
                        f"passed: {command}" if success else f"failed: {command}"
                    )
                    self.working.last_verification_result = self.summary.verification_status
                    if success:
                        self.summary.unresolved_issues = [
                            issue
                            for issue in self.summary.unresolved_issues
                            if not issue.startswith("Verification failed:")
                        ]
                        self.working.unresolved_errors = [
                            issue
                            for issue in self.working.unresolved_errors
                            if not issue.startswith("Verification failed:")
                        ]
                    else:
                        issue = f"Verification failed: {command}: {(result.error or tail)[-1_000:]}"
                        add_unique(self.summary.unresolved_issues, issue)
                        add_unique(self.working.unresolved_errors, issue, limit=20)
                        add_unique(self.summary.next_actions, f"Diagnose and rerun {command}")

        elif call.name == "git_diff" and result.ok:
            self.last_verification_step = step
            self.last_verification_success = True
            self.summary.verification_status = "inspected current git diff"

        if not result.ok:
            issue = f"{call.name} failed: {result.error or 'unknown error'}"
            self.working.last_error = issue
            add_unique(self.working.unresolved_errors, issue, limit=20)
            add_unique(self.summary.unresolved_issues, issue)

        context_result = ToolResult(
            ok=result.ok,
            output=context_output,
            error=result.error,
            metadata=metadata,
            error_code=result.error_code,
            retryable=result.retryable,
        )
        self.append_tool_message(context_result.as_tool_message(call.id), step=step)

    def clear_recent_conversation(self) -> None:
        """Clear raw working messages while preserving fixed and structured memory."""
        self.messages.clear()

    def task_specification(self) -> str:
        if self.fixed is None:
            raise ValueError("Memory has no original task")
        updates = "\n".join(f"- {item}" for item in self.task_updates) or "- none"
        safety = "\n".join(f"- {item}" for item in self.fixed.safety_rules)
        completion = "\n".join(f"- {item}" for item in self.fixed.completion_rules)
        return (
            f"Original task (preserve verbatim):\n{self.fixed.original_task}\n\n"
            f"Subsequent user requirements:\n{updates}\n\n"
            f"Workspace: {self.fixed.workspace}\n"
            f"Maximum agent steps per request: {self.fixed.max_steps}\n\n"
            f"Safety rules:\n{safety}\n\n"
            f"Completion rules:\n{completion}"
        )


def add_unique(items: list[str], value: str, *, limit: int = 100) -> None:
    if value and value not in items:
        items.append(value)
        items[:] = items[-limit:]
