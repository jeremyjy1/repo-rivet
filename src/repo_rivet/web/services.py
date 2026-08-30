"""Application services used by HTTP routes; routers never mutate Agent memory directly."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from repo_rivet.approval.models import ApprovalMode
from repo_rivet.config import load_config
from repo_rivet.editing.document import TextDocument
from repo_rivet.llm.base import ReasoningEffort
from repo_rivet.planning.models import WorkflowMode
from repo_rivet.planning.policy import AutoPlanMode
from repo_rivet.planning.runtime import PlanRuntime
from repo_rivet.reasoning.policy import ReasoningPolicyMode
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.session.models import SessionMetadata, SessionStatus
from repo_rivet.session.store import FileSessionStore
from repo_rivet.skills.registry import SkillRegistry
from repo_rivet.tools.git import GitDiffArguments, GitDiffTool
from repo_rivet.web.events import read_events
from repo_rivet.web.runtime_manager import RuntimeManager, run_view

_IGNORED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
_ACTIVE_RUN_STATUSES = {"queued", "running", "awaiting_approval"}
_AUTOMATIC_SESSION_NAME_MAX_CHARS = 48


def automatic_session_name(task: str) -> str:
    """Derive a stable, immediate conversation title from the first user request."""
    lines = [" ".join(line.split()) for line in task.splitlines() if line.strip()]
    candidate = (lines[0] if lines else "New conversation").strip("#>*-` ")
    if not candidate:
        candidate = "New conversation"
    if len(candidate) <= _AUTOMATIC_SESSION_NAME_MAX_CHARS:
        return candidate
    return candidate[: _AUTOMATIC_SESSION_NAME_MAX_CHARS - 1].rstrip() + "…"


class AgentQueryService:
    def __init__(self, workspace: Path, config_path: Path, sessions: FileSessionStore) -> None:
        self.workspace = workspace.resolve()
        self.config_path = config_path
        self.sessions = sessions
        self.paths = WorkspacePathPolicy(self.workspace)

    def bootstrap(self, manager: RuntimeManager) -> dict[str, Any]:
        active = self.sessions.get_active(self.workspace)
        config = load_config(self.config_path)
        return {
            "workspace": str(self.workspace),
            "active_session_id": active.session_id if active else None,
            "sessions": [
                self._metadata(item)
                for item in self.sessions.list_sessions(workspace=self.workspace, limit=100)
            ],
            "settings": {
                "model": config.api.model,
                "base_url": str(config.api.base_url),
                "context_limit": config.api.context_window_tokens,
                "approval_mode": config.approval.mode.value,
                "auto_plan": (manager.default_auto_plan or config.planning.auto_plan).value,
                "auto_plan_llm": config.planning.llm.enabled,
                "reasoning_policy": manager.default_reasoning_policy.value,
                "reasoning_effort": manager.default_reasoning_effort,
                "reasoning_supported_efforts": list(
                    config.api.reasoning_supported_efforts
                ),
            },
            "capabilities": {"terminal": False, "websocket": False, "sse": True},
        }

    def session(self, session_id: str, manager: RuntimeManager) -> dict[str, Any]:
        loaded = self.sessions.load(session_id)
        messages = [
            {
                "role": item.role,
                "content": item.content,
                "name": item.name,
                "step": item.step,
            }
            for item in loaded.memory.messages
            if item.role != "system" and item.content
        ]
        event_count = len(read_events(loaded.store.events_path, loaded.metadata.session_id))
        return {
            **self._metadata(loaded.metadata),
            "messages": messages,
            "workflow_mode": loaded.memory.workflow_mode.value,
            "approval_mode": (
                loaded.memory.approval_mode_override.value
                if loaded.memory.approval_mode_override
                else None
            ),
            "active_skill": (
                loaded.memory.active_skill.model_dump(mode="json")
                if loaded.memory.active_skill
                else None
            ),
            "plan": (
                loaded.memory.plan_artifact.model_dump(mode="json")
                if loaded.memory.plan_artifact
                else None
            ),
            "verification": {
                key: value.model_dump(mode="json")
                for key, value in loaded.memory.verification_results.items()
            },
            "modified_files": sorted(loaded.memory.modified_files),
            "workspace_revision": loaded.memory.workspace_revision,
            "last_event_seq": event_count,
            "run": run_view(manager.get(loaded.metadata.session_id)),
        }

    def files(self, path: str = ".") -> list[dict[str, Any]]:
        root = self.paths.resolve(path)
        if not root.is_dir():
            raise ValueError("Requested path is not a directory")
        entries: list[dict[str, Any]] = []
        ordered = sorted(
            root.iterdir(),
            key=lambda value: (not value.is_dir(), value.name.lower()),
        )
        for item in ordered:
            if item.name in _IGNORED_DIRS:
                continue
            relative = item.relative_to(self.workspace).as_posix()
            entries.append(
                {
                    "name": item.name,
                    "path": relative,
                    "kind": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
            if len(entries) >= 500:
                break
        return entries

    def file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict[str, Any]:
        relative = self.paths.relative(path).as_posix()
        document = TextDocument.load(self.paths.resolve(relative))
        start = max(1, start_line)
        end = min(document.total_lines, end_line or min(document.total_lines, start + 399))
        lines = document.lines[start - 1 : end]
        return {
            "path": relative,
            "start_line": start,
            "end_line": end,
            "total_lines": document.total_lines,
            "content": "\n".join(lines),
            "snapshot_tag": document.to_snapshot(relative_path=relative).display_tag,
        }

    def diff(self, path: str = ".") -> dict[str, str]:
        result = GitDiffTool(self.paths).run(GitDiffArguments(path=path))
        if not result.ok:
            raise ValueError(result.error or "Could not inspect Git diff")
        return {"path": path, "diff": result.raw_output or result.output}

    def skills(self) -> list[dict[str, Any]]:
        registry = SkillRegistry(
            system_root=Path(__file__).resolve().parents[1] / "builtin_skills",
            global_root=self.sessions.root / "skills",
        )
        return [
            {
                "id": item.qualified_id,
                "name": item.manifest.name,
                "description": item.manifest.description,
                "version": item.version,
                "source": item.source.value,
            }
            for item in registry.discover()
        ]

    @staticmethod
    def _metadata(item: SessionMetadata) -> dict[str, Any]:
        return {
            "session_id": item.session_id,
            "short_id": item.short_id,
            "name": item.name,
            "task_preview": item.task_preview,
            "status": item.status.value,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
            "step": item.step,
            "parent_session_id": item.parent_session_id,
        }


class AgentCommandService:
    def __init__(
        self,
        workspace: Path,
        sessions: FileSessionStore,
        manager: RuntimeManager,
    ) -> None:
        self.workspace = workspace.resolve()
        self.sessions = sessions
        self.manager = manager

    def create_session(self, *, name: str | None = None, task: str = "") -> SessionMetadata:
        return self.sessions.create(workspace=self.workspace, task=task, name=name).metadata

    def use_session(self, session_id: str) -> SessionMetadata:
        return self.sessions.set_active(self.workspace, session_id)

    def fork_session(self, session_id: str, *, name: str | None = None) -> SessionMetadata:
        return self.sessions.fork(session_id, name=name, set_active=True).metadata

    def archive_session(self, session_id: str) -> SessionMetadata:
        return self.sessions.archive(session_id)

    def delete_session(self, session_id: str) -> dict[str, object]:
        resolved_id = self.sessions.resolve_id(session_id)
        run = self.manager.get(resolved_id)
        if run is not None and run.status in {
            "queued",
            "running",
            "awaiting_approval",
            "stopping",
        }:
            raise ValueError("Stop the active run before deleting this conversation")
        self.sessions.purge(resolved_id)
        return {
            "deleted": True,
            "session_id": resolved_id,
            "permanent": True,
        }

    async def submit(
        self,
        session_id: str,
        *,
        task: str,
        mode: WorkflowMode,
        approval_mode: ApprovalMode | None,
        skill: str | None,
        no_skills: bool,
        auto_plan: AutoPlanMode | None,
        clear_skill: bool = False,
        reasoning_policy: ReasoningPolicyMode | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        delivery: Literal["redirect", "queue"] = "redirect",
    ) -> dict[str, Any]:
        resolved_id = self.sessions.resolve_id(session_id)
        self.sessions.set_active(self.workspace, resolved_id)
        self.sessions.ensure_resumable(self.sessions.read_metadata(resolved_id))
        current_run = self.manager.get(resolved_id)
        if current_run is not None and current_run.status in _ACTIVE_RUN_STATUSES:
            self.manager.update_settings(
                resolved_id,
                mode=mode,
                approval_mode=(
                    approval_mode
                    or current_run.approval_mode
                    or ApprovalMode.SAFE_AUTO
                ),
                auto_plan=auto_plan or current_run.auto_plan or AutoPlanMode.ADAPTIVE,
                skill=skill,
                skill_provided=True,
                reasoning_policy=reasoning_policy or current_run.reasoning_policy,
                reasoning_effort=reasoning_effort or current_run.reasoning_effort,
            )
            if delivery == "queue":
                return run_view(await self.manager.enqueue(resolved_id, task)) or {}
            return run_view(await self.manager.steer(resolved_id, task)) or {}
        loaded = self.sessions.load(resolved_id)
        if loaded.metadata.name == "untitled" and not loaded.metadata.task_preview:
            self.sessions.rename(resolved_id, automatic_session_name(task))
        if loaded.memory.workflow_mode == WorkflowMode.PLAN_READY and mode == WorkflowMode.EXECUTE:
            raise ValueError(
                "A plan is waiting for review. Execute it through the plan approval action, "
                "or revise, inspect, or cancel it first."
            )
        run = await self.manager.start(
            session_id=resolved_id,
            task=task,
            mode=mode,
            approval_mode=approval_mode,
            skill=skill,
            clear_skill=clear_skill,
            no_skills=no_skills,
            auto_plan=auto_plan,
            reasoning_policy=reasoning_policy,
            reasoning_effort=reasoning_effort,
        )
        return run_view(run) or {}

    async def stop(self, session_id: str) -> dict[str, Any]:
        return run_view(await self.manager.stop(self.sessions.resolve_id(session_id))) or {}

    def resolve_approval(
        self,
        session_id: str,
        *,
        request_id: str,
        state_version: int,
        action: str,
        guidance: str | None,
    ) -> dict[str, Any]:
        run = self.manager.get(self.sessions.resolve_id(session_id))
        if run is None:
            raise ValueError("This session has no active run")
        decision = run.approver.resolve(
            request_id=request_id,
            state_version=state_version,
            action=action,  # type: ignore[arg-type]
            guidance=guidance,
        )
        return decision.model_dump(mode="json", exclude={"request_fingerprint"})

    async def execute_plan(self, session_id: str) -> dict[str, Any]:
        resolved_id = self.sessions.resolve_id(session_id)
        self.sessions.set_active(self.workspace, resolved_id)
        self.sessions.ensure_resumable(self.sessions.read_metadata(resolved_id))

        loaded_settings = self.sessions.load(resolved_id)
        selected_skill = (
            loaded_settings.memory.active_skill.id
            if loaded_settings.memory.active_skill is not None
            else None
        )

        def prepare() -> None:
            loaded = self.sessions.load(resolved_id)
            runtime = PlanRuntime(WorkspacePathPolicy(self.workspace))
            runtime.bind(loaded.memory)
            artifact = runtime.approve()
            loaded.store.save_state(loaded.memory, status=SessionStatus.EXECUTING.value)
            loaded.store.log("plan_approved", plan_id=artifact.plan_id)

        run = await self.manager.start(
            session_id=resolved_id,
            task="Execute the approved plan from its current step, verify it, and finish.",
            mode=WorkflowMode.EXECUTE,
            skill=selected_skill,
            auto_plan=AutoPlanMode.OFF,
            prepare=prepare,
        )
        return run_view(run) or {}

    def cancel_plan(self, session_id: str) -> None:
        self._ensure_idle(session_id)
        loaded = self.sessions.load(session_id)
        runtime = PlanRuntime(WorkspacePathPolicy(self.workspace))
        runtime.bind(loaded.memory)
        runtime.cancel()
        loaded.store.save_state(loaded.memory, status=SessionStatus.PAUSED.value)
        loaded.store.log("plan_cancelled")

    def set_approval_mode(self, session_id: str, mode: ApprovalMode) -> None:
        self._ensure_idle(session_id)
        loaded = self.sessions.load(session_id)
        loaded.memory.approval_mode_override = mode
        loaded.store.save_state(loaded.memory, status=loaded.memory.status)

    def set_runtime_settings(
        self,
        session_id: str,
        *,
        mode: WorkflowMode | None,
        approval_mode: ApprovalMode | None,
        auto_plan: AutoPlanMode | None,
        skill: str | None,
        skill_provided: bool,
        reasoning_policy: ReasoningPolicyMode | None,
        reasoning_effort: ReasoningEffort | None,
    ) -> dict[str, Any]:
        resolved_id = self.sessions.resolve_id(session_id)
        run = self.manager.update_settings(
            resolved_id,
            mode=mode,
            approval_mode=approval_mode,
            auto_plan=auto_plan,
            skill=skill,
            skill_provided=skill_provided,
            reasoning_policy=reasoning_policy,
            reasoning_effort=reasoning_effort,
        )
        return run_view(run) or {}

    def clear_skill(self, session_id: str) -> None:
        self._ensure_idle(session_id)
        loaded = self.sessions.load(session_id)
        loaded.memory.active_skill = None
        loaded.store.save_state(loaded.memory, status=loaded.memory.status)

    def _ensure_idle(self, session_id: str) -> None:
        run = self.manager.get(self.sessions.resolve_id(session_id))
        if run is not None and run.status in {"queued", "running", "stopping"}:
            raise ValueError("Session settings cannot change while a run is active")
