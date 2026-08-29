"""FastAPI composition root for the local RepoRivet GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from repo_rivet.approval.models import ApprovalMode
from repo_rivet.config import load_config
from repo_rivet.planning.models import WorkflowMode
from repo_rivet.planning.policy import AutoPlanMode
from repo_rivet.session.store import FileSessionStore
from repo_rivet.web.auth import LocalAuth
from repo_rivet.web.events import EventBroker, event_stream, read_event_page
from repo_rivet.web.runtime_manager import RuntimeManager
from repo_rivet.web.services import AgentCommandService, AgentQueryService


class BootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=20, max_length=200)


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, max_length=100)
    task: str = Field(default="", max_length=10_000)


class ForkSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, max_length=100)


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(min_length=1, max_length=50_000)
    mode: WorkflowMode = WorkflowMode.EXECUTE
    approval_mode: ApprovalMode | None = None
    skill: str | None = Field(default=None, max_length=100)
    no_skills: bool = False
    auto_plan: AutoPlanMode | None = None


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    state_version: int = Field(ge=1)
    action: Literal["allow_once", "allow_session", "deny", "stop"]
    guidance: str | None = Field(default=None, max_length=1_000)


class PlanPromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str = Field(min_length=1, max_length=20_000)


class ApprovalModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: ApprovalMode


@dataclass(slots=True)
class WebContext:
    auth: LocalAuth
    broker: EventBroker
    manager: RuntimeManager
    queries: AgentQueryService
    commands: AgentCommandService
    sessions: FileSessionStore
    static_dir: Path


def create_app(
    *,
    workspace: str | Path,
    config_path: str | Path = "reporivet.toml",
    expected_origin: str = "http://127.0.0.1:8000",
    bootstrap_token: str | None = None,
    static_dir: Path | None = None,
    max_steps: int = 30,
    max_seconds: float = 600,
    reasoning: str | None = None,
    default_approval_mode: ApprovalMode | None = None,
    default_auto_plan: AutoPlanMode | None = None,
    default_skill: str | None = None,
    no_skills: bool = False,
) -> FastAPI:
    resolved_workspace = Path(workspace).expanduser().resolve(strict=True)
    resolved_config = Path(config_path).expanduser()
    if not resolved_config.is_absolute():
        resolved_config = (Path.cwd() / resolved_config).resolve()
    config = load_config(resolved_config)
    sessions = FileSessionStore(secrets=(config.api.api_key.get_secret_value(),))
    broker = EventBroker()
    auth = LocalAuth(expected_origin=expected_origin)
    if bootstrap_token is not None:
        auth.bootstrap_token = bootstrap_token
    manager = RuntimeManager(
        workspace=resolved_workspace,
        config_path=resolved_config,
        broker=broker,
        max_steps=max_steps,
        max_seconds=max_seconds,
        reasoning=reasoning,
        default_approval_mode=default_approval_mode,
        default_auto_plan=default_auto_plan,
        default_skill=default_skill,
        no_skills=no_skills,
    )
    queries = AgentQueryService(resolved_workspace, resolved_config, sessions)
    commands = AgentCommandService(resolved_workspace, sessions, manager)
    distribution = static_dir or Path(__file__).resolve().parent.parent / "web_dist"
    context = WebContext(auth, broker, manager, queries, commands, sessions, distribution)

    app = FastAPI(title="RepoRivet GUI", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.reporivet = context

    @app.middleware("http")
    async def secure_local_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        host = request.headers.get("host", "")
        if host != context.auth.expected_host:
            return JSONResponse(status_code=400, content={"detail": "Invalid Host header"})
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        return response

    def require_session(request: Request) -> None:
        context.auth.require_session(request)

    def require_write(request: Request) -> None:
        context.auth.require_write(request)

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.post("/api/v1/auth/bootstrap")
    def bootstrap(payload: BootstrapRequest, response: Response) -> dict[str, str]:
        return {"csrf_token": context.auth.bootstrap(payload.token, response)}

    @app.get("/api/v1/bootstrap", dependencies=[Depends(require_session)])
    def application_bootstrap() -> dict[str, object]:
        return context.queries.bootstrap(context.manager)

    @app.get("/api/v1/sessions", dependencies=[Depends(require_session)])
    def sessions_list() -> list[dict[str, object]]:
        return [
            context.queries._metadata(item)
            for item in context.sessions.list_sessions(workspace=resolved_workspace, limit=100)
        ]

    @app.post("/api/v1/sessions", dependencies=[Depends(require_write)])
    def session_create(payload: CreateSessionRequest) -> dict[str, object]:
        return context.queries._metadata(
            context.commands.create_session(name=payload.name, task=payload.task)
        )

    @app.get("/api/v1/sessions/{session_id}", dependencies=[Depends(require_session)])
    def session_get(session_id: str) -> dict[str, object]:
        return context.queries.session(session_id, context.manager)

    @app.post("/api/v1/sessions/{session_id}/use", dependencies=[Depends(require_write)])
    def session_use(session_id: str) -> dict[str, object]:
        return context.queries._metadata(context.commands.use_session(session_id))

    @app.post("/api/v1/sessions/{session_id}/fork", dependencies=[Depends(require_write)])
    def session_fork(session_id: str, payload: ForkSessionRequest) -> dict[str, object]:
        return context.queries._metadata(
            context.commands.fork_session(session_id, name=payload.name)
        )

    @app.post("/api/v1/sessions/{session_id}/archive", dependencies=[Depends(require_write)])
    def session_archive(session_id: str) -> dict[str, object]:
        return context.queries._metadata(context.commands.archive_session(session_id))

    @app.post("/api/v1/sessions/{session_id}/runs", dependencies=[Depends(require_write)])
    async def run_start(session_id: str, payload: SubmitRequest) -> dict[str, object]:
        return await context.commands.submit(
            session_id,
            task=payload.task,
            mode=payload.mode,
            approval_mode=payload.approval_mode,
            skill=payload.skill,
            no_skills=payload.no_skills,
            auto_plan=payload.auto_plan,
        )

    @app.post("/api/v1/sessions/{session_id}/stop", dependencies=[Depends(require_write)])
    async def run_stop(session_id: str) -> dict[str, object]:
        return await context.commands.stop(session_id)

    @app.post(
        "/api/v1/sessions/{session_id}/approvals/decision",
        dependencies=[Depends(require_write)],
    )
    def approval_decide(session_id: str, payload: ApprovalDecisionRequest) -> dict[str, object]:
        return context.commands.resolve_approval(
            session_id,
            request_id=payload.request_id,
            state_version=payload.state_version,
            action=payload.action,
            guidance=payload.guidance,
        )

    @app.post("/api/v1/sessions/{session_id}/plan/execute", dependencies=[Depends(require_write)])
    async def plan_execute(session_id: str) -> dict[str, object]:
        return await context.commands.execute_plan(session_id)

    @app.post("/api/v1/sessions/{session_id}/plan/revise", dependencies=[Depends(require_write)])
    async def plan_revise(session_id: str, payload: PlanPromptRequest) -> dict[str, object]:
        return await context.commands.submit(
            session_id,
            task=f"Revise the current plan according to this feedback: {payload.instruction}",
            mode=WorkflowMode.PLANNING,
            approval_mode=None,
            skill=None,
            no_skills=False,
        )

    @app.post("/api/v1/sessions/{session_id}/plan/inspect", dependencies=[Depends(require_write)])
    async def plan_inspect(session_id: str, payload: PlanPromptRequest) -> dict[str, object]:
        return await context.commands.submit(
            session_id,
            task=f"Continue read-only planning and inspection: {payload.instruction}",
            mode=WorkflowMode.PLANNING,
            approval_mode=None,
            skill=None,
            no_skills=False,
        )

    @app.post("/api/v1/sessions/{session_id}/plan/cancel", dependencies=[Depends(require_write)])
    def plan_cancel(session_id: str) -> dict[str, bool]:
        context.commands.cancel_plan(session_id)
        return {"cancelled": True}

    @app.put(
        "/api/v1/sessions/{session_id}/approval-mode",
        dependencies=[Depends(require_write)],
    )
    def approval_mode(session_id: str, payload: ApprovalModeRequest) -> dict[str, str]:
        context.commands.set_approval_mode(session_id, payload.mode)
        return {"mode": payload.mode.value}

    @app.delete("/api/v1/sessions/{session_id}/skill", dependencies=[Depends(require_write)])
    def skill_clear(session_id: str) -> dict[str, bool]:
        context.commands.clear_skill(session_id)
        return {"cleared": True}

    @app.get("/api/v1/skills", dependencies=[Depends(require_session)])
    def skills() -> list[dict[str, object]]:
        return context.queries.skills()

    @app.get("/api/v1/workspace/files", dependencies=[Depends(require_session)])
    def workspace_files(path: str = Query(default=".")) -> list[dict[str, object]]:
        return context.queries.files(path)

    @app.get("/api/v1/workspace/file", dependencies=[Depends(require_session)])
    def workspace_file(
        path: str,
        start_line: int = Query(default=1, ge=1),
        end_line: int | None = Query(default=None, ge=1),
    ) -> dict[str, object]:
        return context.queries.file(path, start_line, end_line)

    @app.get("/api/v1/workspace/diff", dependencies=[Depends(require_session)])
    def workspace_diff(path: str = Query(default=".")) -> dict[str, str]:
        return context.queries.diff(path)

    @app.get("/api/v1/sessions/{session_id}/events", dependencies=[Depends(require_session)])
    async def events(
        request: Request,
        session_id: str,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        loaded = context.sessions.load(session_id)
        header_cursor = request.headers.get("last-event-id")
        if header_cursor:
            try:
                after = max(after, int(header_cursor))
            except ValueError:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid Last-Event-ID") from None
        return StreamingResponse(
            event_stream(
                path=loaded.store.events_path,
                session_id=loaded.metadata.session_id,
                broker=context.broker,
                after=after,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(
        "/api/v1/sessions/{session_id}/events/history",
        dependencies=[Depends(require_session)],
    )
    def event_history(
        session_id: str,
        before: int | None = Query(default=None, ge=1),
        limit: int = Query(default=240, ge=1, le=500),
    ) -> dict[str, object]:
        loaded = context.sessions.load(session_id)
        items, has_more = read_event_page(
            loaded.store.events_path,
            loaded.metadata.session_id,
            before=before,
            limit=limit,
        )
        return {
            "items": [item.as_dict() for item in items],
            "has_more": has_more,
        }

    assets = context.static_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        del path
        index = context.static_dir / "index.html"
        if not index.is_file():
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "GUI assets are not built. Run `npm install && npm run build` in frontend/.",
            )
        return FileResponse(index)

    return app
