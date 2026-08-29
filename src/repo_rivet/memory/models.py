"""Typed fixed, working, summary, file, and command memory models."""

import json
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repo_rivet.approval.models import ApprovalMode, OperationClass
from repo_rivet.memory.token_estimator import ApproximateTokenEstimator
from repo_rivet.reasoning.models import ObservationEvent, ReasoningEvent
from repo_rivet.tools.base import ToolCall, ToolResult
from repo_rivet.verification.models import (
    FinalAssessment,
    ModelErrorRecord,
    ProcessObservation,
    VerificationPlan,
    VerificationResult,
    VerificationStatus,
)


class MemoryConfig(BaseModel):
    """Context budget and compaction thresholds for one session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recent_message_limit: int = Field(default=10, ge=4, le=30)
    max_context_tokens: int = Field(default=24_000, ge=1_000)
    active_prompt_limit: int = Field(default=65_536, ge=1_000)
    reserved_output_tokens: int = Field(default=4_096, ge=100)
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
    # Provider continuation state is needed in memory for the current request chain, but hidden
    # reasoning must never become durable session memory.
    reasoning_content: str | None = Field(default=None, exclude=True, repr=False)
    ephemeral: bool = Field(default=False, exclude=True, repr=False)
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    step: int = 0

    @classmethod
    def from_chat_message(cls, message: dict[str, Any], *, step: int = 0) -> "Message":
        return cls(
            role=message["role"],
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            tool_call_id=message.get("tool_call_id"),
            name=message.get("name"),
            tool_calls=message.get("tool_calls"),
            step=step,
        )

    def as_chat_message(self) -> dict[str, Any]:
        content = self._model_visible_content()
        message: dict[str, Any] = {"role": self.role, "content": content}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            message["name"] = self.name
        if self.tool_calls is not None:
            message["tool_calls"] = self.tool_calls
        return message

    def is_valid_provider_message(self) -> bool:
        """Return whether an assistant message has provider-visible content or tool calls."""
        if self.role != "assistant":
            return True
        return bool(
            (self.content and self.content.strip())
            or self.tool_calls
            or (self.reasoning_content and self.reasoning_content.strip())
        )

    def is_valid_durable_message(self) -> bool:
        """Return whether persistence will retain a valid provider-visible message."""
        if self.role != "assistant":
            return True
        return bool((self.content and self.content.strip()) or self.tool_calls)

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


class ArtifactRecord(BaseModel):
    """Durable provenance for an output created by an approved session operation."""

    model_config = ConfigDict(extra="forbid")

    path: str
    artifact_type: Literal["executable", "build_output", "generated"]
    created_by_session: str
    created_by_request: str
    producer_operation: OperationClass
    source_paths: list[str] = Field(default_factory=list)
    content_sha256: str
    workspace_revision: int = Field(ge=0)


class MemoryState(BaseModel):
    """Serializable state for one RepoRivet session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    config: MemoryConfig = Field(default_factory=MemoryConfig)
    fixed: FixedMemory | None = None
    task_updates: list[str] = Field(default_factory=list)
    context_checkpoint: str | None = None
    messages: list[Message] = Field(default_factory=list)
    summary: ConversationSummary = Field(default_factory=ConversationSummary)
    working: WorkingMemory = Field(default_factory=WorkingMemory)
    file_memories: dict[str, FileMemory] = Field(default_factory=dict)
    current_snapshots: dict[str, str] = Field(default_factory=dict)
    invalidated_files: set[str] = Field(default_factory=set)
    modified_files: set[str] = Field(default_factory=set)
    workspace_revision: int = Field(default=0, ge=0)
    verification_plan: VerificationPlan | None = None
    verification_results: dict[str, VerificationResult] = Field(default_factory=dict)
    verification_plan_recovery_attempts: int = Field(default=0, ge=0)
    candidate_final_assessment: FinalAssessment | None = None
    last_model_error: ModelErrorRecord | None = None
    provider_requires_reasoning_content: bool = False
    command_outputs: list[CommandOutputMemory] = Field(default_factory=list)
    process_observations: list[ProcessObservation] = Field(default_factory=list)
    artifact_registry: dict[str, ArtifactRecord] = Field(default_factory=dict)
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
    last_agent_outcome: Literal["success", "incomplete", "blocked", "stopped", "error"] | None = (
        None
    )
    status: str = "ready"

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_verification_fields(cls, value: Any) -> Any:
        """Discard command-name verification state from pre-plan session snapshots."""
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        migrated.pop("last_file_change_step", None)
        migrated.pop("last_verification_step", None)
        migrated.pop("last_verification_success", None)
        if "workspace_revision" not in migrated and migrated.get("modified_files"):
            migrated["workspace_revision"] = 1
        return migrated

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

    def begin_task_scope(self) -> None:
        """Start fresh completion evidence after a previously completed request."""
        if self.last_agent_outcome != "success":
            self.last_agent_outcome = None
            return
        self.modified_files.clear()
        self.verification_plan = None
        self.verification_results.clear()
        self.verification_plan_recovery_attempts = 0
        self.candidate_final_assessment = None
        self.last_model_error = None
        self.reflection_required = False
        self.working.recent_modified_files.clear()
        self.working.last_verification_result = None
        self.last_agent_outcome = None

    def repair_invalid_assistant_messages(self) -> int:
        """Remove legacy empty assistant messages rejected by compatible providers."""
        before = len(self.messages)
        self.messages[:] = [
            message for message in self.messages if message.is_valid_provider_message()
        ]
        return before - len(self.messages)

    def repair_interrupted_tool_history(
        self,
        *,
        error_for: Callable[[str, str], str] | None = None,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Close unfinished tool groups in place and discard orphan tool results."""
        repaired: list[Message] = []
        pending: dict[str, str] = {}
        pending_step = 0
        missing: list[tuple[str, str]] = []
        orphan_results: list[str] = []
        seen_call_ids: set[str] = set()

        def close_pending() -> None:
            for call_id, name in pending.items():
                error = (
                    error_for(call_id, name)
                    if error_for is not None
                    else (
                        "The previous run ended before this tool result was recorded. "
                        "The tool was not retried automatically."
                    )
                )
                repaired.append(
                    Message(
                        role="tool",
                        tool_call_id=call_id,
                        content=json.dumps(
                            {
                                "ok": False,
                                "error": error,
                                "error_code": "interrupted_tool_call",
                            },
                            ensure_ascii=False,
                        ),
                        step=pending_step,
                    )
                )
                missing.append((call_id, name))
            pending.clear()

        for message in self.messages:
            if pending:
                if message.role == "tool":
                    call_id = message.tool_call_id
                    if isinstance(call_id, str) and call_id in pending:
                        repaired.append(message)
                        del pending[call_id]
                    else:
                        orphan_results.append(str(call_id or "missing"))
                    continue
                close_pending()

            if message.role == "assistant" and message.tool_calls:
                valid_calls: list[dict[str, Any]] = []
                for raw_call in message.tool_calls:
                    if not isinstance(raw_call, dict):
                        continue
                    call_id = raw_call.get("id")
                    function = raw_call.get("function")
                    name = function.get("name") if isinstance(function, dict) else None
                    if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                        continue
                    valid_calls.append(raw_call)
                    pending[call_id] = str(name or "unknown tool")
                    seen_call_ids.add(call_id)
                if valid_calls:
                    repaired.append(message.model_copy(update={"tool_calls": valid_calls}))
                    pending_step = message.step
                elif message.content and message.content.strip():
                    repaired.append(message.model_copy(update={"tool_calls": None}))
                continue

            if message.role == "tool":
                orphan_results.append(str(message.tool_call_id or "missing"))
                continue
            repaired.append(message)

        if pending:
            close_pending()
        self.messages[:] = repaired
        return missing, orphan_results

    def append_assistant(
        self,
        message: dict[str, Any],
        *,
        step: int,
        ephemeral: bool = False,
    ) -> None:
        value = Message.from_chat_message(message, step=step)
        value.ephemeral = ephemeral
        self.messages.append(value)

    def append_ephemeral_system(self, content: str, *, step: int) -> None:
        self.messages.append(Message(role="system", content=content, step=step, ephemeral=True))

    def discard_ephemeral_messages(self) -> None:
        self.messages[:] = [message for message in self.messages if not message.ephemeral]

    def discard_provider_state(self) -> None:
        """Drop hidden continuation data when the current agent run terminates."""
        self.discard_ephemeral_messages()
        for message in self.messages:
            message.reasoning_content = None

    def append_tool_message(self, message: dict[str, Any], *, step: int) -> None:
        self.messages.append(Message.from_chat_message(message, step=step))

    def record_tool_result(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        step: int,
        full_output_path: str | None = None,
        append_message: bool = True,
    ) -> None:
        """Update structured memory and append a bounded, linked tool observation."""
        self.tool_event_step = max(self.tool_event_step, step)
        metadata = dict(result.metadata or {})
        process_value = metadata.get("process_observation")
        if isinstance(process_value, dict):
            process_observation = ProcessObservation.model_validate(process_value)
            if full_output_path:
                process_observation.stdout_ref = full_output_path
                process_observation.stderr_ref = full_output_path
            self.process_observations.append(process_observation)
            self.process_observations[:] = self.process_observations[-20:]
        context_output = result.output
        path = metadata.get("path") or call.arguments.get("path")

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
                snapshot_id = metadata.get("snapshot_id")
                if isinstance(snapshot_id, str):
                    self.current_snapshots.pop(path, None)
                    self.current_snapshots[path] = snapshot_id

        elif call.name in {"write_file", "edit_file"} and result.ok and isinstance(path, str):
            self.file_memories.pop(path, None)
            self.invalidated_files.add(path)
            self.modified_files.add(path)
            self.workspace_revision += 1
            for check_id, verification in list(self.verification_results.items()):
                if verification.status != VerificationStatus.STALE:
                    self.verification_results[check_id] = verification.model_copy(
                        update={"status": VerificationStatus.STALE}
                    )
            self.summary.verification_status = "stale after workspace modification"
            self.working.last_verification_result = self.summary.verification_status
            add_unique(self.summary.files_modified, path)
            add_unique(self.summary.completed_actions, f"Modified {path} with {call.name}")
            add_unique(self.working.recent_modified_files, path, limit=20)
            snapshot_id = metadata.get("new_snapshot_id") or metadata.get("snapshot_id")
            if isinstance(snapshot_id, str):
                self.current_snapshots.pop(path, None)
                self.current_snapshots[path] = snapshot_id
            context_output = (
                f"{context_output}\nThe previous snapshot for {path} is invalid. "
                "Use the returned new snapshot only for displayed changed ranges; reread other "
                "ranges before editing them."
            )

        elif call.name in {"run_command", "run_verification"}:
            command = call.arguments.get("command") or metadata.get("command")
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

                verification_value = metadata.get("verification_result")
                if call.name == "run_verification" and isinstance(verification_value, dict):
                    verification = VerificationResult.model_validate(verification_value)
                    if full_output_path:
                        verification.stdout_ref = full_output_path
                        verification.stderr_ref = full_output_path
                    self.verification_results[verification.check_id] = verification
                    self.summary.verification_status = (
                        f"{verification.status.value}: {verification.check_id}"
                    )
                    self.working.last_verification_result = self.summary.verification_status
                    if verification.status == VerificationStatus.PASSED:
                        self.summary.unresolved_issues = [
                            issue
                            for issue in self.summary.unresolved_issues
                            if not issue.startswith(f"Verification {verification.check_id} ")
                        ]
                    else:
                        issue = (
                            f"Verification {verification.check_id} "
                            f"{verification.status.value}: {'; '.join(verification.reasons)}"
                        )
                        add_unique(self.summary.unresolved_issues, issue)
                        add_unique(self.working.unresolved_errors, issue, limit=20)

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
        if append_message:
            self.append_tool_message(context_result.as_tool_message(call.id), step=step)

    def clear_recent_conversation(self) -> None:
        """Clear raw working messages while preserving fixed and structured memory."""
        self.messages.clear()
        self.context_checkpoint = None

    def task_specification(self) -> str:
        """Render only session-stable task information for the prompt prefix."""
        if self.fixed is None:
            raise ValueError("Memory has no original task")
        safety = "\n".join(f"- {item}" for item in self.fixed.safety_rules)
        completion = "\n".join(f"- {item}" for item in self.fixed.completion_rules)
        return (
            f"Original task (preserve verbatim):\n{self.fixed.original_task}\n\n"
            f"Workspace: {self.fixed.workspace}\n"
            f"Maximum agent steps per request: {self.fixed.max_steps}\n\n"
            f"Safety rules:\n{safety}\n\n"
            f"Completion rules:\n{completion}"
        )


def add_unique(items: list[str], value: str, *, limit: int = 100) -> None:
    if value and value not in items:
        items.append(value)
        items[:] = items[-limit:]
