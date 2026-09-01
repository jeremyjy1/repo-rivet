"""Provider-visible delegation and structured child-report tools."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from repo_rivet.memory.models import MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.subagents.models import DelegateTaskArguments, SubagentReport
from repo_rivet.tools.base import BaseTool, ToolResult

_MAX_EVIDENCE_CHARS = 12_000


class DelegationRunner(Protocol):
    def delegate(self, arguments: DelegateTaskArguments) -> ToolResult: ...


class DelegateTaskTool(BaseTool[DelegateTaskArguments]):
    name = "delegate_task"
    description = (
        "Delegate one bounded, read-only investigation to a scoped child agent and wait for its "
        "validated evidence report. Use for independent repository exploration, test-failure "
        "analysis, or diff review; do not use for edits, commands, or an already-obvious next step."
    )
    arguments_type = DelegateTaskArguments
    wait_kind = "subagent_results"

    def __init__(self, runner: DelegationRunner) -> None:
        self.runner = runner

    def run(self, arguments: DelegateTaskArguments) -> ToolResult:
        return self.runner.delegate(arguments)


class SubmitSubagentReportArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: SubagentReport


class SubagentReportCollector:
    def __init__(self, delegation_id: str) -> None:
        self.delegation_id = delegation_id
        self.report: SubagentReport | None = None

    def submit(self, report: SubagentReport) -> None:
        if self.report is not None:
            raise ValueError("A subagent may submit only one final report")
        if report.delegation_id != self.delegation_id:
            raise ValueError("Subagent report delegation_id does not match the active delegation")
        self.report = report


class SubmitSubagentReportTool(BaseTool[SubmitSubagentReportArguments]):
    name = "submit_subagent_report"
    description = (
        "Submit the single final structured report for this child run. Every material finding "
        "must cite real evidence returned by a permitted tool. This ends the child run."
    )
    arguments_type = SubmitSubagentReportArguments

    def __init__(self, collector: SubagentReportCollector) -> None:
        self.collector = collector

    def run(self, arguments: SubmitSubagentReportArguments) -> ToolResult:
        self.collector.submit(arguments.report)
        return ToolResult(
            ok=True,
            output="Subagent report submitted for parent validation.",
            metadata={"delegation_id": arguments.report.delegation_id},
        )


class ReadToolOutputArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_ref: str = Field(min_length=1)


class ReadToolOutputTool(BaseTool[ReadToolOutputArguments]):
    """Read bounded delegated parent evidence or evidence produced by this child."""

    name = "read_tool_output"
    description = (
        "Read the persisted full output for either a parent observation listed in evidence_refs "
        "or an observation produced earlier in this child run. The result is bounded and cannot "
        "access arbitrary session files."
    )
    arguments_type = ReadToolOutputArguments

    def __init__(
        self,
        parent_store: MemoryStore,
        parent_memory: MemoryState,
        allowed_evidence_refs: list[str],
        *,
        child_store: MemoryStore | None = None,
        child_memory: MemoryState | None = None,
    ) -> None:
        self.parent_store = parent_store
        self.parent_memory = parent_memory
        self.allowed_evidence_refs = frozenset(allowed_evidence_refs)
        self.child_store = child_store
        self.child_memory = child_memory

    def run(self, arguments: ReadToolOutputArguments) -> ToolResult:
        reference = arguments.evidence_ref
        store = self.parent_store
        memory = self.parent_memory
        if reference not in self.allowed_evidence_refs:
            child_observations = (
                self.child_memory.observation_events if self.child_memory is not None else []
            )
            if not any(event.event_id == reference for event in child_observations):
                raise ValueError(f"Evidence was not delegated to this subagent: {reference}")
            if self.child_store is None or self.child_memory is None:
                raise ValueError(f"Child evidence store is unavailable: {reference}")
            store = self.child_store
            memory = self.child_memory
        observation = next(
            (
                event
                for event in memory.observation_events
                if event.event_id == reference
            ),
            None,
        )
        if observation is None:
            raise ValueError(f"Delegated evidence is not a tool observation: {reference}")
        if not observation.output_ref:
            raise ValueError(f"Delegated observation has no persisted full output: {reference}")
        output_path = (store.session_dir / observation.output_ref).resolve()
        allowed_roots = (
            store.command_output_dir.resolve(),
            store.file_snapshot_dir.resolve(),
        )
        if not any(output_path.is_relative_to(root) for root in allowed_roots):
            raise ValueError(f"Invalid persisted output reference: {reference}")
        try:
            content = output_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ValueError(f"Persisted output is unavailable: {reference}") from None
        output, truncated = _bounded_output(content)
        return ToolResult(
            ok=True,
            output=output,
            metadata={
                "evidence_ref": reference,
                "tool": observation.tool_name,
                "truncated": truncated,
                "original_chars": len(content),
            },
        )


class ReadVerificationResultArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(min_length=1)


class ReadVerificationResultTool(BaseTool[ReadVerificationResultArguments]):
    """Read an already-recorded verification fact without rerunning it."""

    name = "read_verification_result"
    description = (
        "Read one existing parent verification result explicitly listed in evidence_refs. "
        "This never executes the verification command."
    )
    arguments_type = ReadVerificationResultArguments

    def __init__(self, parent_memory: MemoryState, allowed_evidence_refs: list[str]) -> None:
        self.parent_memory = parent_memory
        self.allowed_evidence_refs = frozenset(allowed_evidence_refs)

    def run(self, arguments: ReadVerificationResultArguments) -> ToolResult:
        check_id = arguments.check_id
        if check_id not in self.allowed_evidence_refs:
            raise ValueError(f"Verification evidence was not delegated: {check_id}")
        result = self.parent_memory.verification_results.get(check_id)
        if result is None:
            raise ValueError(f"Verification result is unavailable: {check_id}")
        payload = result.model_dump(mode="json", exclude={"stdout_ref", "stderr_ref"})
        return ToolResult(
            ok=True,
            output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            metadata={"evidence_ref": check_id, "status": result.status.value},
        )


def _bounded_output(content: str) -> tuple[str, bool]:
    if len(content) <= _MAX_EVIDENCE_CHARS:
        return content, False
    edge = _MAX_EVIDENCE_CHARS // 2
    omitted = len(content) - (edge * 2)
    return (
        f"{content[:edge]}\n\n... {omitted:,} characters omitted ...\n\n{content[-edge:]}",
        True,
    )
