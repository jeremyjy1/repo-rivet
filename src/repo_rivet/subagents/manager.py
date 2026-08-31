"""Construct, run, validate, persist, and reuse bounded child runtimes."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from repo_rivet.agent.controller import AgentController
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy
from repo_rivet.editing.runtime import EditingRuntime
from repo_rivet.llm.base import ModelClient
from repo_rivet.memory.context_manager import ContextManager
from repo_rivet.memory.models import MemoryConfig, MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.planning.policy import AutoPlanMode, AutoPlanPolicy
from repo_rivet.reasoning.manager import ReasoningManager
from repo_rivet.reasoning.models import ReasoningConfig
from repo_rivet.storage.atomic_write import atomic_write_json
from repo_rivet.storage.event_sink import CompositeEventSink
from repo_rivet.subagents.models import (
    DelegateTaskArguments,
    DelegationRequest,
    SubagentRecord,
    SubagentReport,
    SubagentStatus,
    ValidatedSubagentReport,
)
from repo_rivet.subagents.policy import (
    ScopedWorkspacePathPolicy,
    normalize_scope,
    profile_runtime_config,
)
from repo_rivet.subagents.tools import (
    ReadToolOutputTool,
    ReadVerificationResultTool,
    SubagentReportCollector,
    SubmitSubagentReportTool,
)
from repo_rivet.subagents.validator import SubagentResultValidator
from repo_rivet.tools.base import BaseTool, ToolResult
from repo_rivet.tools.filesystem import ListFilesTool, ReadFileTool, SearchTextTool
from repo_rivet.tools.git import GitDiffTool
from repo_rivet.tools.registry import ToolRegistry


class EventSink(Protocol):
    def log(self, event_type: str, **data: Any) -> None: ...


ModelClientFactory = Callable[[EventSink], ModelClient]


SUBAGENT_SYSTEM_PROMPT = """You are a bounded, read-only RepoRivet subagent.
You are not the parent agent and must not broaden or complete the parent task.
Use only the tools provided to inspect the explicitly allowed paths. You cannot modify files,
run commands, request approval, ask the user, or spawn another subagent. Treat the objective and
deliverable as a strict contract. Every material finding must cite an evidence_ref returned by a
tool. Do not invent evidence, snapshots, files, or verification results. Finish only by calling
submit_subagent_report exactly once. A normal text response is not a valid report.
"""


class SubagentManager:
    """Own read-only child runs beneath one durable parent session."""

    def __init__(
        self,
        *,
        workspace: Path,
        parent_store: MemoryStore,
        model_client_factory: ModelClientFactory,
        event_logger: EventSink,
        secrets: tuple[str, ...] = (),
        max_concurrency: int = 2,
    ) -> None:
        self.workspace = workspace.resolve()
        self.parent_store = parent_store
        self.model_client_factory = model_client_factory
        self.event_logger = event_logger
        self.secrets = secrets
        self.max_concurrency = max_concurrency
        self.root = parent_store.session_dir / "subagents"
        self.validator = SubagentResultValidator(self.workspace)
        self._parent_memory: MemoryState | None = None
        self._active: set[str] = set()
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def bind(self, memory: MemoryState) -> None:
        self._parent_memory = memory

    def delegate(self, arguments: DelegateTaskArguments) -> ToolResult:
        parent_memory = self._memory()
        request = self._request(arguments, parent_memory)
        self.event_logger.log(
            "delegation_requested",
            delegation_id=request.delegation_id,
            profile=request.profile.value,
            objective=request.objective,
            scope_paths=request.scope_paths,
        )
        reusable = self._find_reusable(request, parent_memory)
        if reusable is not None:
            record, validated = reusable
            self.event_logger.log(
                "subagent_report_accepted",
                subagent_id=record.subagent_id,
                delegation_id=record.delegation_id,
                profile=record.profile.value,
                reused=True,
                freshness=validated.freshness,
                summary=validated.report.summary,
            )
            return self._result(record, validated, reused=True)

        semantic_key = self._semantic_key(request)
        subagent_id = f"subagent-{uuid4().hex[:12]}"
        with self._lock:
            existing = self._inflight.get(semantic_key)
            if existing is None and len(self._active) >= self.max_concurrency:
                return ToolResult(
                    ok=False,
                    output="",
                    error=f"Subagent concurrency limit reached ({self.max_concurrency})",
                    error_code="subagent_concurrency_limit",
                    retryable=True,
                )
            if existing is None:
                completion = threading.Event()
                self._inflight[semantic_key] = completion
                self._active.add(subagent_id)
                owner = True
            else:
                completion = existing
                owner = False
        if not owner:
            completion.wait(timeout=request.max_runtime_seconds + 5)
            reusable = self._find_reusable(request, parent_memory)
            if reusable is not None:
                record, validated = reusable
                return self._result(record, validated, reused=True)
            return ToolResult(
                ok=False,
                output="",
                error="Matching subagent run ended without a reusable report",
                error_code="subagent_join_failed",
                retryable=True,
            )
        try:
            return self._run_child(subagent_id, request, parent_memory)
        finally:
            with self._lock:
                self._active.discard(subagent_id)
                completed = self._inflight.pop(semantic_key, None)
                if completed is not None:
                    completed.set()

    def _run_child(
        self,
        subagent_id: str,
        request: DelegationRequest,
        parent_memory: MemoryState,
    ) -> ToolResult:
        child_directory = self.root / subagent_id
        child_store = MemoryStore(child_directory, secrets=self.secrets)
        record = SubagentRecord(
            subagent_id=subagent_id,
            delegation_id=request.delegation_id,
            semantic_key=self._semantic_key(request),
            parent_run_id=request.parent_run_id,
            profile=request.profile,
            status=SubagentStatus.CREATED,
            base_workspace_revision=request.base_workspace_revision,
            scope_paths=request.scope_paths,
        )
        self._save_request(child_directory, request)
        self._save_record(child_directory, record)
        self.event_logger.log(
            "subagent_created",
            subagent_id=subagent_id,
            delegation_id=request.delegation_id,
            profile=request.profile.value,
        )

        record.status = SubagentStatus.RUNNING
        record.started_at = datetime.now(UTC)
        self._save_record(child_directory, record)
        self.event_logger.log(
            "subagent_started",
            subagent_id=subagent_id,
            delegation_id=request.delegation_id,
            profile=request.profile.value,
            scope_paths=request.scope_paths,
        )
        collector = SubagentReportCollector(request.delegation_id)
        registry = self._registry(request, child_directory, collector)
        child_events = CompositeEventSink(child_store)
        config = profile_runtime_config(
            request.profile,
            request.scope_paths,
            max_model_calls=request.max_model_calls,
            max_tool_calls=request.max_tool_calls,
            max_runtime_seconds=request.max_runtime_seconds,
        )
        memory = MemoryState(
            session_id=subagent_id,
            config=MemoryConfig(
                recent_message_limit=8,
                max_context_tokens=12_000,
                active_prompt_limit=12_000,
                reserved_output_tokens=2_048,
                reserved_tool_result_tokens=512,
                max_tool_output_chars=8_000,
            ),
            workspace_revision=request.base_workspace_revision,
        )
        controller = AgentController(
            model_client=self.model_client_factory(child_events),
            tool_registry=registry,
            context_manager=ContextManager(),
            termination_policy=TerminationPolicy(
                TerminationConfig(
                    max_steps=config.max_model_calls,
                    max_tool_calls=config.max_tool_calls,
                    max_seconds=config.max_runtime_seconds,
                    max_consecutive_failures=2,
                    max_consecutive_protocol_failures=2,
                    max_empty_model_responses=2,
                    max_consecutive_length_responses=2,
                )
            ),
            event_logger=child_events,
            memory_store=child_store,
            reasoning_manager=ReasoningManager(
                ReasoningConfig(enabled=False, recent_event_limit=20),
                secrets=self.secrets,
            ),
            auto_plan_policy=AutoPlanPolicy(AutoPlanMode.OFF),
            system_prompt=SUBAGENT_SYSTEM_PROMPT,
            safety_rules=(
                "The child runtime is read-only and path-scoped.",
                "It cannot request approval, run commands, or delegate recursively.",
            ),
            completion_rules=("Submit exactly one validated subagent report.",),
            terminal_tool_names=frozenset({"submit_subagent_report"}),
        )
        try:
            result = controller.run(self._task_prompt(request, parent_memory), memory=memory)
            if memory.runtime is not None:
                record.child_run_id = memory.runtime.run_id
            report = collector.report
            if result.status != "success" or report is None:
                raise ValueError(result.reason or "Subagent ended without a structured report")
            self.event_logger.log(
                "subagent_report_submitted",
                subagent_id=subagent_id,
                delegation_id=request.delegation_id,
                status=report.status,
            )
            record.status = SubagentStatus.REPORT_READY
            report_event_id = f"subreport-{uuid4().hex[:12]}"
            record.report_event_id = report_event_id
            atomic_write_json(
                child_directory / "report.json",
                report.model_dump(mode="json"),
            )
            self._save_record(child_directory, record)
            validated = self.validator.validate(
                report,
                request=request,
                child_memory=memory,
                child_directory=child_directory,
                parent_memory=parent_memory,
            )
            if validated.freshness == "stale":
                record.status = SubagentStatus.STALE
                record.error = "Referenced file snapshots changed before integration"
                self.event_logger.log(
                    "subagent_marked_stale",
                    subagent_id=subagent_id,
                    delegation_id=request.delegation_id,
                    stale_paths=validated.stale_paths,
                )
                raise ValueError(record.error)
            record.status = (
                SubagentStatus.BLOCKED
                if report.status in {"blocked", "inconclusive"}
                else SubagentStatus.ACCEPTED
            )
            record.finished_at = datetime.now(UTC)
            self._save_record(child_directory, record)
            atomic_write_json(
                child_directory / "report.json",
                validated.report.model_dump(mode="json"),
            )
            self.event_logger.log(
                "subagent_report_validated",
                subagent_id=subagent_id,
                delegation_id=request.delegation_id,
                freshness=validated.freshness,
                finding_count=len(validated.report.findings),
            )
            self.event_logger.log(
                "subagent_report_accepted",
                subagent_id=subagent_id,
                delegation_id=request.delegation_id,
                profile=request.profile.value,
                reused=False,
                freshness=validated.freshness,
                summary=validated.report.summary,
                status=validated.report.status,
            )
            return self._result(record, validated, reused=False)
        except Exception as error:
            record.status = (
                record.status
                if record.status in {SubagentStatus.STALE, SubagentStatus.BLOCKED}
                else SubagentStatus.ERROR
            )
            record.finished_at = datetime.now(UTC)
            record.error = str(error)
            self._save_record(child_directory, record)
            self.event_logger.log(
                "subagent_blocked",
                subagent_id=subagent_id,
                delegation_id=request.delegation_id,
                reason=str(error),
            )
            return ToolResult(
                ok=False,
                output="",
                error=f"Subagent did not produce an acceptable report: {error}",
                error_code="subagent_report_invalid",
                retryable=False,
                metadata={"subagent_id": subagent_id, "delegation_id": request.delegation_id},
            )

    def _registry(
        self,
        request: DelegationRequest,
        child_directory: Path,
        collector: SubagentReportCollector,
    ) -> ToolRegistry:
        policy = ScopedWorkspacePathPolicy(
            self.workspace,
            allowed_paths=request.scope_paths,
            excluded_paths=request.excluded_paths,
        )
        editing = EditingRuntime(policy, snapshot_dir=child_directory / "snapshots")
        available: dict[str, BaseTool[Any]] = {
            "list_files": ListFilesTool(policy),
            "read_file": ReadFileTool(policy, editing),
            "search_text": SearchTextTool(policy, editing),
            "git_diff": GitDiffTool(policy),
            "read_tool_output": ReadToolOutputTool(
                self.parent_store,
                self._memory(),
                request.evidence_refs,
            ),
            "read_verification_result": ReadVerificationResultTool(
                self._memory(),
                request.evidence_refs,
            ),
            "submit_subagent_report": SubmitSubagentReportTool(collector),
        }
        config = profile_runtime_config(request.profile, request.scope_paths)
        return ToolRegistry(
            [tool for name, tool in available.items() if name in config.allowed_tools],
            workspace=self.workspace,
        )

    def _request(
        self,
        arguments: DelegateTaskArguments,
        parent_memory: MemoryState,
    ) -> DelegationRequest:
        scope = normalize_scope(self.workspace, arguments.scope_paths)
        self._validate_parent_evidence(arguments.evidence_refs, scope, parent_memory)
        runtime = parent_memory.runtime
        parent_run_id = (
            runtime.run_id if runtime is not None else f"parent-{parent_memory.session_id}"
        )
        plan_revision = runtime.revisions.plan if runtime is not None else 0
        return DelegationRequest(
            delegation_id=f"delegation-{uuid4().hex[:12]}",
            parent_run_id=parent_run_id,
            profile=arguments.profile,
            objective=" ".join(arguments.objective.split()),
            deliverable=" ".join(arguments.deliverable.split()),
            scope_paths=scope,
            evidence_refs=arguments.evidence_refs,
            constraints=[" ".join(value.split()) for value in arguments.constraints],
            base_workspace_revision=parent_memory.workspace_revision,
            base_plan_revision=plan_revision,
            join_policy=arguments.join_policy,
        )

    def _validate_parent_evidence(
        self,
        evidence_refs: list[str],
        scope_paths: list[str],
        parent_memory: MemoryState,
    ) -> None:
        policy = ScopedWorkspacePathPolicy(self.workspace, allowed_paths=scope_paths)
        observations = {event.event_id: event for event in parent_memory.observation_events}
        known = {
            *observations,
            *parent_memory.verification_results,
        }
        missing = set(evidence_refs) - known
        if missing:
            raise ValueError(
                "Delegation references unknown evidence: " + ", ".join(sorted(missing))
            )
        for reference in evidence_refs:
            observation = observations.get(reference)
            if observation is None:
                continue
            for path in observation.affected_paths:
                policy.resolve(path)

    def _task_prompt(self, request: DelegationRequest, parent_memory: MemoryState) -> str:
        evidence = self._evidence_context(request.evidence_refs, parent_memory)
        return json.dumps(
            {
                "delegation_id": request.delegation_id,
                "profile": request.profile.value,
                "objective": request.objective,
                "deliverable": request.deliverable,
                "allowed_paths": request.scope_paths,
                "constraints": request.constraints,
                "base_workspace_revision": request.base_workspace_revision,
                "base_plan_revision": request.base_plan_revision,
                "provided_evidence": evidence,
                "instruction": (
                    "Investigate only this scope. End by calling submit_subagent_report with "
                    "the exact delegation_id and base_workspace_revision."
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _evidence_context(refs: list[str], memory: MemoryState) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        observations = {event.event_id: event for event in memory.observation_events}
        for reference in refs:
            if reference in observations:
                observation = observations[reference]
                values.append(
                    {
                        "ref": reference,
                        "kind": "observation",
                        "tool": observation.tool_name,
                        "summary": observation.result_summary,
                        "paths": observation.affected_paths,
                    }
                )
            elif reference in memory.verification_results:
                result = memory.verification_results[reference]
                values.append(
                    {
                        "ref": reference,
                        "kind": "verification",
                        "status": result.status.value,
                        "exit_code": result.exit_code,
                        "reasons": result.reasons,
                    }
                )
        return values

    def _find_reusable(
        self,
        request: DelegationRequest,
        parent_memory: MemoryState,
    ) -> tuple[SubagentRecord, ValidatedSubagentReport] | None:
        semantic_key = self._semantic_key(request)
        if not self.root.is_dir():
            return None
        for directory in sorted(self.root.iterdir(), reverse=True):
            record_path = directory / "record.json"
            report_path = directory / "report.json"
            if not record_path.is_file() or not report_path.is_file():
                continue
            try:
                record = SubagentRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
                if record.semantic_key != semantic_key or record.status not in {
                    SubagentStatus.ACCEPTED,
                    SubagentStatus.BLOCKED,
                    SubagentStatus.REPORT_READY,
                }:
                    continue
                stored_request = DelegationRequest.model_validate_json(
                    (directory / "request.json").read_text(encoding="utf-8")
                )
                report = SubagentReport.model_validate_json(report_path.read_text(encoding="utf-8"))
                child_memory = MemoryStore(directory, secrets=self.secrets).load_state()
                validated = self.validator.validate(
                    report,
                    request=stored_request,
                    child_memory=child_memory,
                    child_directory=directory,
                    parent_memory=parent_memory,
                )
            except (OSError, ValueError):
                continue
            if validated.freshness == "fresh":
                return record, validated
            record.status = SubagentStatus.STALE
            record.error = "Referenced file snapshots changed"
            self._save_record(directory, record)
        return None

    @staticmethod
    def _semantic_key(request: DelegationRequest) -> str:
        payload = {
            "profile": request.profile.value,
            "objective": request.objective.casefold(),
            "deliverable": request.deliverable.casefold(),
            "scope_paths": sorted(request.scope_paths),
            "base_workspace_revision": request.base_workspace_revision,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    @staticmethod
    def _result(
        record: SubagentRecord,
        validated: ValidatedSubagentReport,
        *,
        reused: bool,
    ) -> ToolResult:
        report = validated.report
        output = json.dumps(
            {
                "subagent_id": record.subagent_id,
                "profile": record.profile.value,
                "status": report.status,
                "summary": report.summary,
                "findings": [item.model_dump(mode="json") for item in report.findings],
                "recommended_actions": report.recommended_actions,
                "unknowns": report.unknowns,
                "freshness": validated.freshness,
                "reused": reused,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return ToolResult(
            ok=True,
            output=output,
            metadata={
                "subagent_id": record.subagent_id,
                "delegation_id": record.delegation_id,
                "profile": record.profile.value,
                "report_status": report.status,
                "freshness": validated.freshness,
                "reused": reused,
                "evidence_refs": sorted(
                    {reference for item in report.findings for reference in item.evidence_refs}
                ),
            },
        )

    @staticmethod
    def _save_record(directory: Path, record: SubagentRecord) -> None:
        atomic_write_json(directory / "record.json", record.model_dump(mode="json"))

    @staticmethod
    def _save_request(directory: Path, request: DelegationRequest) -> None:
        atomic_write_json(directory / "request.json", request.model_dump(mode="json"))

    def _memory(self) -> MemoryState:
        if self._parent_memory is None:
            raise RuntimeError("Subagent manager is not bound to parent memory")
        return self._parent_memory
