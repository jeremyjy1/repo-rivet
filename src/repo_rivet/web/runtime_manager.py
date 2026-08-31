"""Async orchestration around the synchronous Agent Core."""

from __future__ import annotations

import argparse
import asyncio
import io
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from rich.console import Console

from repo_rivet.agent.state import SessionState
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy
from repo_rivet.approval.models import ApprovalMode
from repo_rivet.cli import _build_runtime, _idle_session_status
from repo_rivet.llm.base import ReasoningEffort
from repo_rivet.planning.classifier import OpenAIPlanClassifier
from repo_rivet.planning.models import WorkflowMode
from repo_rivet.planning.policy import AutoPlanMode, AutoPlanPolicy
from repo_rivet.reasoning.policy import ReasoningPolicyMode
from repo_rivet.session.models import SessionStatus
from repo_rivet.storage.event_sink import EventSink
from repo_rivet.web.approvals import WebHumanApprover
from repo_rivet.web.events import BrokerEventSink, EventBroker


class InterruptibleTerminationPolicy(TerminationPolicy):
    def __init__(self, stop_event: threading.Event, config: TerminationConfig) -> None:
        super().__init__(config)
        self.stop_event = stop_event

    def check(
        self,
        state: SessionState,
        *,
        now: float | None = None,
        include_step_limit: bool = True,
    ) -> str | None:
        if self.stop_event.is_set():
            return "interrupted by user"
        return super().check(state, now=now, include_step_limit=include_step_limit)


@dataclass(frozen=True, slots=True)
class RunInput:
    instruction: str
    delivery: Literal["redirect", "queue"]


class RunInputMailbox:
    """Thread-safe ordered input shared by HTTP handlers and the synchronous agent."""

    def __init__(self, *, max_pending: int = 20) -> None:
        self._messages: list[RunInput] = []
        self._lock = threading.Lock()
        self._max_pending = max_pending
        self._accepting = True

    def submit(
        self,
        instruction: str,
        *,
        delivery: Literal["redirect", "queue"],
    ) -> None:
        normalized = instruction.strip()
        if not normalized:
            raise ValueError("Direction must not be empty")
        with self._lock:
            if not self._accepting:
                raise ValueError("The current run is finishing; submit again when it completes")
            if len(self._messages) >= self._max_pending:
                raise ValueError("Too many pending messages for this run")
            self._messages.append(RunInput(normalized, delivery))

    def has_redirect(self) -> bool:
        with self._lock:
            return any(message.delivery == "redirect" for message in self._messages)

    def drain_redirects(self) -> list[str]:
        with self._lock:
            redirects = [
                message.instruction for message in self._messages if message.delivery == "redirect"
            ]
            self._messages[:] = [
                message for message in self._messages if message.delivery != "redirect"
            ]
            return redirects

    def pop_or_close(self) -> RunInput | None:
        """Take a late follow-up or atomically stop accepting redirects."""
        with self._lock:
            if self._messages:
                return self._messages.pop(0)
            self._accepting = False
            return None


@dataclass(slots=True)
class ActiveRun:
    run_id: str
    session_id: str
    workspace: Path
    task: str
    mode: WorkflowMode
    approval_mode: ApprovalMode | None = None
    auto_plan: AutoPlanMode | None = None
    skill: str | None = None
    no_skills: bool = False
    reasoning_policy: ReasoningPolicyMode = ReasoningPolicyMode.ADAPTIVE
    reasoning_effort: ReasoningEffort = "max"
    stop_event: threading.Event = field(default_factory=threading.Event)
    inputs: RunInputMailbox = field(default_factory=RunInputMailbox)
    approver: WebHumanApprover = field(default_factory=WebHumanApprover)
    task_handle: asyncio.Task[None] | None = None
    event_logger: EventSink | None = None
    input_event_lock: threading.Lock = field(default_factory=threading.Lock)
    settings_lock: threading.Lock = field(default_factory=threading.Lock)
    pending_approval_mode: ApprovalMode | None = None
    pending_skill_update: bool = False
    pending_planning_mode: bool = False
    pending_reasoning_policy: ReasoningPolicyMode | None = None
    model_client: Any | None = None
    pending_input_events: list[tuple[str, Literal["redirect", "queue"]]] = field(
        default_factory=list
    )
    status: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None


class RuntimeManager:
    """Guarantee one active run per session and per workspace."""

    def __init__(
        self,
        *,
        workspace: Path,
        config_path: Path,
        broker: EventBroker,
        max_steps: int = 30,
        max_seconds: float = 600,
        reasoning: str | None = None,
        default_approval_mode: ApprovalMode | None = None,
        default_auto_plan: AutoPlanMode | None = None,
        default_reasoning_policy: ReasoningPolicyMode = ReasoningPolicyMode.ADAPTIVE,
        default_reasoning_effort: ReasoningEffort = "max",
        default_skill: str | None = None,
        no_skills: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.config_path = config_path
        self.broker = broker
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.reasoning = reasoning
        self.default_approval_mode = default_approval_mode
        self.default_auto_plan = default_auto_plan
        self.default_reasoning_policy = default_reasoning_policy
        self.default_reasoning_effort = default_reasoning_effort
        self.default_skill = default_skill
        self.no_skills = no_skills
        self._runs: dict[str, ActiveRun] = {}
        self._workspace_runs: dict[Path, str] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        *,
        session_id: str,
        task: str,
        mode: WorkflowMode,
        approval_mode: ApprovalMode | None = None,
        skill: str | None = None,
        clear_skill: bool = False,
        no_skills: bool = False,
        auto_plan: AutoPlanMode | None = None,
        reasoning_policy: ReasoningPolicyMode | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        prepare: Callable[[], None] | None = None,
    ) -> ActiveRun:
        normalized = task.strip()
        if not normalized:
            raise ValueError("Task must not be empty")
        async with self._lock:
            existing = self._runs.get(session_id)
            if existing is not None and existing.status in {
                "queued",
                "running",
                "awaiting_approval",
                "stopping",
            }:
                raise ValueError("This session already has an active run")
            workspace_owner = self._workspace_runs.get(self.workspace)
            if workspace_owner is not None:
                raise ValueError("Another session is already changing this workspace")
            if prepare is not None:
                prepare()
            run = ActiveRun(
                run_id=f"run-{uuid4().hex[:12]}",
                session_id=session_id,
                workspace=self.workspace,
                task=normalized,
                mode=mode,
                approval_mode=approval_mode or self.default_approval_mode,
                auto_plan=auto_plan or self.default_auto_plan,
                skill=(None if clear_skill else skill or self.default_skill),
                no_skills=no_skills or self.no_skills,
                reasoning_policy=reasoning_policy or self.default_reasoning_policy,
                reasoning_effort=reasoning_effort or self.default_reasoning_effort,
            )
            self._runs[session_id] = run
            self._workspace_runs[self.workspace] = session_id
            loop = asyncio.get_running_loop()
            run.task_handle = asyncio.create_task(
                self._worker(
                    run,
                    loop=loop,
                    approval_mode=approval_mode,
                    skill=skill,
                    no_skills=no_skills,
                    auto_plan=auto_plan,
                )
            )
            return run

    def update_settings(
        self,
        session_id: str,
        *,
        mode: WorkflowMode | None = None,
        approval_mode: ApprovalMode | None = None,
        auto_plan: AutoPlanMode | None = None,
        skill: str | None = None,
        skill_provided: bool = False,
        reasoning_policy: ReasoningPolicyMode | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> ActiveRun:
        """Update an active run; live-safe settings apply at the next controller boundary."""
        run = self._runs.get(session_id)
        if run is None or run.status not in {"queued", "running", "awaiting_approval"}:
            raise ValueError("This session has no active run to configure")
        with run.settings_lock:
            previous_effort = run.reasoning_effort
            previous_policy = run.reasoning_policy
            if mode is not None:
                run.mode = mode
                if mode == WorkflowMode.PLANNING:
                    run.pending_planning_mode = True
            if approval_mode is not None and approval_mode != run.approval_mode:
                run.approval_mode = approval_mode
                run.pending_approval_mode = approval_mode
            if auto_plan is not None:
                run.auto_plan = auto_plan
            if skill_provided:
                run.skill = skill
                run.pending_skill_update = True
            if reasoning_effort is not None:
                run.reasoning_effort = reasoning_effort
            if reasoning_policy is not None and reasoning_policy != run.reasoning_policy:
                run.reasoning_policy = reasoning_policy
                run.pending_reasoning_policy = reasoning_policy
            model_client = run.model_client
            event_logger = run.event_logger
        setter = getattr(model_client, "set_reasoning_effort_ceiling", None)
        if reasoning_effort is not None and callable(setter):
            setter(reasoning_effort)
        if event_logger is not None:
            event_logger.log(
                "runtime_settings_changed",
                mode=run.mode.value,
                approval_mode=run.approval_mode.value if run.approval_mode else None,
                auto_plan=run.auto_plan.value if run.auto_plan else None,
                skill=run.skill,
                reasoning_policy=run.reasoning_policy.value,
                reasoning_effort=run.reasoning_effort,
                reasoning_effort_changed=(
                    reasoning_effort is not None and previous_effort != reasoning_effort
                ),
                reasoning_policy_changed=(
                    reasoning_policy is not None and previous_policy != reasoning_policy
                ),
                application="live_safe_boundary",
            )
        self.broker.notify(session_id)
        return run

    async def stop(self, session_id: str) -> ActiveRun:
        run = self._runs.get(session_id)
        if run is None or run.status not in {"queued", "running", "awaiting_approval"}:
            raise ValueError("This session has no active run")
        run.stop_event.set()
        run.approver.abort_pending()
        run.status = "stopping"
        return run

    async def steer(self, session_id: str, instruction: str) -> ActiveRun:
        run = self._runs.get(session_id)
        if run is None or run.status not in {"queued", "running", "awaiting_approval"}:
            raise ValueError("This session has no active run to redirect")
        run.inputs.submit(instruction, delivery="redirect")
        self._announce_input(run, instruction, delivery="redirect")
        run.approver.redirect_pending(instruction)
        self.broker.notify(session_id)
        return run

    async def enqueue(self, session_id: str, instruction: str) -> ActiveRun:
        run = self._runs.get(session_id)
        if run is None or run.status not in {"queued", "running", "awaiting_approval"}:
            raise ValueError("This session has no active run for a follow-up")
        run.inputs.submit(instruction, delivery="queue")
        self._announce_input(run, instruction, delivery="queue")
        self.broker.notify(session_id)
        return run

    @staticmethod
    def _announce_input(
        run: ActiveRun,
        instruction: str,
        *,
        delivery: Literal["redirect", "queue"],
    ) -> None:
        with run.input_event_lock:
            event_logger = run.event_logger
            if event_logger is None:
                run.pending_input_events.append((instruction.strip(), delivery))
                return
            event_logger.log("user_input", task=instruction.strip(), delivery=delivery)

    def get(self, session_id: str) -> ActiveRun | None:
        return self._runs.get(session_id)

    @staticmethod
    def _drain_live_settings(run: ActiveRun) -> dict[str, str]:
        with run.settings_lock:
            mode = run.pending_approval_mode
            run.pending_approval_mode = None
            skill_update = run.pending_skill_update
            run.pending_skill_update = False
            planning_mode = run.pending_planning_mode
            run.pending_planning_mode = False
            reasoning_policy = run.pending_reasoning_policy
            run.pending_reasoning_policy = None
            skill = run.skill
        updates: dict[str, str] = {}
        if mode is not None:
            updates["approval_mode"] = mode.value
        if skill_update:
            updates["skill"] = skill or ""
        if planning_mode:
            updates["workflow_mode"] = WorkflowMode.PLANNING.value
        if reasoning_policy is not None:
            updates["reasoning_policy"] = reasoning_policy.value
        return updates

    async def _worker(
        self,
        run: ActiveRun,
        *,
        loop: asyncio.AbstractEventLoop,
        approval_mode: ApprovalMode | None,
        skill: str | None,
        no_skills: bool,
        auto_plan: AutoPlanMode | None,
    ) -> None:
        run.status = "running"
        try:
            await asyncio.to_thread(
                self._run_sync,
                run,
                loop,
                approval_mode or self.default_approval_mode,
                skill or self.default_skill,
                no_skills or self.no_skills,
                auto_plan or self.default_auto_plan,
            )
        except Exception as error:  # worker errors must remain queryable by the UI
            run.status = "error"
            run.error = f"{type(error).__name__}: {error}"
        finally:
            async with self._lock:
                if self._workspace_runs.get(run.workspace) == run.session_id:
                    self._workspace_runs.pop(run.workspace, None)
            self.broker.notify(run.session_id)

    def _run_sync(
        self,
        run: ActiveRun,
        loop: asyncio.AbstractEventLoop,
        approval_mode: ApprovalMode | None,
        skill: str | None,
        no_skills: bool,
        auto_plan: AutoPlanMode | None,
    ) -> None:
        arguments = argparse.Namespace(
            command="gui",
            workspace=self.workspace,
            config=self.config_path,
            session=run.session_id,
            max_steps=self.max_steps,
            max_seconds=self.max_seconds,
            reasoning=self.reasoning,
            approval_mode=approval_mode.value if approval_mode else None,
            skill=skill,
            no_skills=no_skills,
            auto_plan=auto_plan.value if auto_plan else None,
        )
        termination = InterruptibleTerminationPolicy(
            run.stop_event,
            TerminationConfig(max_steps=self.max_steps, max_seconds=self.max_seconds),
        )
        sink = BrokerEventSink(run.session_id, self.broker, loop)
        runtime = _build_runtime(
            arguments,
            Console(file=io.StringIO(), force_terminal=False, color_system=None),
            human_approver_override=run.approver,
            additional_event_sinks=(sink,),
            console_events=False,
            termination_policy=termination,
        )
        with run.input_event_lock:
            event_logger = runtime.controller.event_logger
            if event_logger is None:
                raise RuntimeError("Web runtime requires an event logger")
            run.event_logger = event_logger
            pending_input_events = list(run.pending_input_events)
            run.pending_input_events.clear()
            for instruction, delivery in pending_input_events:
                event_logger.log(
                    "user_input",
                    task=instruction,
                    delivery=delivery,
                )
        run.approver.bind_event_logger(runtime.controller.event_logger)
        with run.settings_lock:
            run.model_client = runtime.controller.model_client
            reasoning_effort = run.reasoning_effort
        set_reasoning_effort = getattr(
            runtime.controller.model_client,
            "set_reasoning_effort_ceiling",
            None,
        )
        if callable(set_reasoning_effort):
            set_reasoning_effort(reasoning_effort)
        runtime.controller.set_reasoning_policy_mode(run.reasoning_policy)
        runtime.controller.set_steering_source(run.inputs.drain_redirects)
        runtime.controller.set_runtime_settings_source(lambda: self._drain_live_settings(run))
        set_interrupt_checker = getattr(
            runtime.controller.model_client,
            "set_interrupt_checker",
            None,
        )
        if callable(set_interrupt_checker):
            set_interrupt_checker(run.inputs.has_redirect)
        try:
            task = run.task
            while True:
                with run.settings_lock:
                    mode = run.mode
                    auto_plan_mode = run.auto_plan or runtime.config.planning.auto_plan
                    selected_skill = run.skill
                runtime.controller.auto_plan_policy = AutoPlanPolicy(
                    auto_plan_mode,
                    classifier_confidence_threshold=(
                        runtime.config.planning.llm.confidence_threshold
                    ),
                )
                runtime.controller.plan_classifier = (
                    OpenAIPlanClassifier(
                        runtime.config.api,
                        model=runtime.config.planning.llm.model,
                        timeout_seconds=runtime.config.planning.llm.timeout_seconds,
                    )
                    if auto_plan_mode == AutoPlanMode.ADAPTIVE
                    and runtime.config.planning.llm.enabled
                    else None
                )
                if runtime.skill_runtime is not None:
                    if selected_skill is None:
                        runtime.skill_runtime.clear(runtime.memory)
                    elif (
                        runtime.memory.active_skill is None
                        or runtime.memory.active_skill.id != selected_skill
                    ):
                        runtime.skill_runtime.activate(runtime.memory, selected_skill)
                result = runtime.controller.run(
                    task,
                    memory=runtime.memory,
                    workflow_mode=mode,
                )
                follow_up = run.inputs.pop_or_close()
                if follow_up is None or run.stop_event.is_set():
                    break
                task = follow_up.instruction
            idle = _idle_session_status(runtime.memory)
            runtime.store.save_state(
                runtime.memory,
                status=idle.value,
                agent_step=result.step_count,
            )
            run.result = {
                "status": result.status,
                "summary": result.summary,
                "reason": result.reason,
                "modified_files": list(result.modified_files),
                "step_count": result.step_count,
                "tool_call_count": result.tool_call_count,
                "verification_status": result.verification_status.value,
            }
            run.status = (
                "completed" if result.status in {"success", "plan_ready"} else result.status
            )
            # Publish completion only after the queryable run state is complete. SSE clients
            # refresh immediately when this event arrives and must never observe result=None.
            runtime.store.log(
                "web_run_finished",
                run_id=run.run_id,
                status=result.status,
                reason=result.reason,
            )
        except Exception:
            runtime.memory.status = SessionStatus.FAILED.value
            runtime.store.save_state(runtime.memory, status=SessionStatus.FAILED.value)
            raise
        finally:
            with run.input_event_lock:
                run.event_logger = None
            with run.settings_lock:
                run.model_client = None
            if callable(set_interrupt_checker):
                set_interrupt_checker(None)
            runtime.controller.set_steering_source(None)
            runtime.controller.set_runtime_settings_source(None)
            run.approver.bind_event_logger(None)
            runtime.close()


def run_view(run: ActiveRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    with run.settings_lock:
        settings = {
            "mode": run.mode.value,
            "approval_mode": run.approval_mode.value if run.approval_mode else None,
            "auto_plan": run.auto_plan.value if run.auto_plan else None,
            "skill": run.skill,
            "reasoning_policy": run.reasoning_policy.value,
            "reasoning_effort": run.reasoning_effort,
        }
    pending = run.approver.snapshot()
    status = "awaiting_approval" if pending is not None and run.status == "running" else run.status
    return {
        "run_id": run.run_id,
        "session_id": run.session_id,
        "status": status,
        "mode": run.mode.value,
        "settings": settings,
        "result": run.result,
        "error": run.error,
        "pending_approval": pending,
    }
