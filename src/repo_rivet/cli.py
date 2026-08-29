"""Argparse and Rich command-line interface for RepoRivet."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from repo_rivet import __version__
from repo_rivet.agent.controller import AgentController, AgentResult
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy
from repo_rivet.approval.engine import ApprovalEngine
from repo_rivet.approval.grant_store import ApprovalGrantStore
from repo_rivet.approval.hard_policy import HardSafetyPolicy, HardSafetySettings
from repo_rivet.approval.human_approver import (
    NonInteractiveHumanApprover,
    TerminalHumanApprover,
)
from repo_rivet.approval.llm_reviewer import OpenAIApprovalReviewer
from repo_rivet.approval.models import ApprovalMode, RiskLevel
from repo_rivet.approval.normalizer import RequestNormalizer
from repo_rivet.approval.risk_analyzer import RiskAnalyzer
from repo_rivet.approval.semantic_analyzer import ApprovalFactAnalyzer
from repo_rivet.config import AppConfig, ConfigurationError, load_config
from repo_rivet.context.manager import ContextManager
from repo_rivet.llm.openai_compatible import OpenAICompatibleClient
from repo_rivet.memory.budget_manager import TokenBudgetConfig, TokenBudgetManager
from repo_rivet.memory.compactor import ConversationCompactor
from repo_rivet.memory.models import MemoryConfig, MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.memory.token_calibrator import TokenCalibrationStore
from repo_rivet.memory.token_estimator import create_token_estimator
from repo_rivet.reasoning.manager import ReasoningManager
from repo_rivet.reasoning.models import ReasoningDisplayMode
from repo_rivet.safety.path_policy import PathPolicyError
from repo_rivet.session.errors import SessionError
from repo_rivet.session.lock import SessionLock
from repo_rivet.session.models import SessionMetadata, SessionStatus
from repo_rivet.session.store import FileSessionStore, LoadedSession
from repo_rivet.storage.console_reporter import ConsoleEventReporter
from repo_rivet.storage.event_sink import CompositeEventSink, EventSink
from repo_rivet.storage.terminal_text import escape_terminal_controls
from repo_rivet.tools.registry import ToolRegistry, create_default_registry


class PromptReader(Protocol):
    def __call__(self, prompt: str) -> str:
        """Read one line of interactive input."""
        ...


class ConversationAgent(Protocol):
    def run(
        self,
        task: str,
        *,
        memory: MemoryState | None = None,
    ) -> AgentResult:
        """Run one task with optional prior conversation."""
        ...


class ApprovalModeManager(Protocol):
    mode: ApprovalMode

    def set_mode(self, mode: ApprovalMode) -> None:
        """Apply and persist an approval-mode switch."""
        ...


@dataclass(slots=True)
class Runtime:
    config: AppConfig
    registry: ToolRegistry
    store: MemoryStore
    memory: MemoryState
    controller: AgentController
    session_manager: FileSessionStore
    session_lock: SessionLock
    loaded_existing_session: bool = False

    def close(self) -> None:
        self.session_lock.__exit__(None, None, None)


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line grammar."""
    parser = argparse.ArgumentParser(
        prog="reporivet",
        description="Run a local-first coding agent in a confined workspace.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the coding agent")
    run_parser.add_argument("task", nargs="+", help="Programming task for the agent")
    _add_runtime_arguments(run_parser)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive coding conversation")
    chat_parser.add_argument(
        "initial_task",
        nargs="*",
        help="Optional first request before the interactive prompt",
    )
    _add_runtime_arguments(chat_parser)

    session_parser = subparsers.add_parser("session", help="Manage local conversations")
    session_commands = session_parser.add_subparsers(dest="session_command", required=True)

    list_parser = session_commands.add_parser("list", help="List saved sessions")
    list_parser.add_argument("--workspace", type=Path)
    list_parser.add_argument("--status", choices=[status.value for status in SessionStatus])
    list_parser.add_argument("--all", action="store_true", dest="include_archived")
    list_parser.add_argument("--limit", type=int, default=20)

    show_parser = session_commands.add_parser("show", help="Show session details")
    show_parser.add_argument("session_id")

    current_parser = session_commands.add_parser("current", help="Show the selected session")
    current_parser.add_argument("--workspace", type=Path, default=Path.cwd())

    use_parser = session_commands.add_parser("use", help="Select without running a session")
    use_parser.add_argument("session_id")
    use_parser.add_argument("--workspace", type=Path)

    resume_parser = session_commands.add_parser("resume", help="Resume a session in chat")
    resume_parser.add_argument("session_id", nargs="?")
    resume_parser.add_argument("--workspace", type=Path)
    _add_model_runtime_arguments(resume_parser)

    new_parser = session_commands.add_parser("new", help="Create an empty session")
    new_parser.add_argument("task", nargs="?")
    new_parser.add_argument("--name")
    new_parser.add_argument("--workspace", type=Path, default=Path.cwd())

    rename_parser = session_commands.add_parser("rename", help="Rename a session")
    rename_parser.add_argument("session_id")
    rename_parser.add_argument("name")

    fork_parser = session_commands.add_parser("fork", help="Branch from a saved session")
    fork_parser.add_argument("session_id")
    fork_parser.add_argument("--name")
    fork_parser.add_argument("--use", action="store_true")

    archive_parser = session_commands.add_parser("archive", help="Archive a session")
    archive_parser.add_argument("session_id")

    delete_parser = session_commands.add_parser("delete", help="Move a session to trash")
    delete_parser.add_argument("session_id")
    delete_parser.add_argument("--yes", action="store_true")

    repair_parser = session_commands.add_parser("repair", help="Repair an interrupted session")
    repair_parser.add_argument("session_id")
    return parser


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Target workspace (default: current directory)",
    )
    parser.add_argument("--session", help="Explicit session ID (full or unique short ID)")
    _add_model_runtime_arguments(parser)


def _add_model_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("reporivet.toml"),
        help="Local API configuration file (default: reporivet.toml)",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument(
        "--reasoning",
        choices=[mode.value for mode in ReasoningDisplayMode],
        help="Display structured decision records: off, summary, or trace",
    )
    parser.add_argument(
        "--approval-mode",
        choices=[mode.value for mode in ApprovalMode],
        help="Override the configured tool approval mode",
    )


def cli(
    argv: Sequence[str] | None = None,
    *,
    console: Console | None = None,
    prompt_reader: PromptReader | None = None,
) -> int:
    """Run the CLI and return a process exit code."""
    output = console or Console()
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        return _run_agent(arguments, output)
    if arguments.command == "chat":
        reader = prompt_reader or (lambda prompt: Prompt.ask(prompt, console=output))
        return _chat_agent(
            arguments,
            output,
            reader,
            approval_prompt_reader=prompt_reader,
        )
    if arguments.command == "session":
        return _session_command(arguments, output, prompt_reader=prompt_reader)
    output.print("[red]Unknown command.[/red]")
    return 2


def _run_agent(arguments: argparse.Namespace, console: Console) -> int:
    task = " ".join(arguments.task).strip()
    if not task:
        console.print("[bold red]Task error:[/bold red] Task must not be empty")
        return 2

    try:
        runtime = _build_runtime(arguments, console)
    except (ConfigurationError, PathPolicyError, SessionError, OSError, ValueError) as error:
        console.print(f"[bold red]Configuration error:[/bold red] {error}")
        return 2

    _print_runtime(console, arguments.workspace, runtime)
    try:
        try:
            result = runtime.controller.run(task, memory=runtime.memory)
        except OSError as error:
            runtime.memory.status = SessionStatus.FAILED.value
            runtime.store.save_state(runtime.memory, status=SessionStatus.FAILED.value)
            console.print(f"[bold red]Runtime error:[/bold red] {error}")
            return 1
        _print_result(console, result)
        return 0 if result.status == "success" else 1
    finally:
        runtime.close()


def _chat_agent(
    arguments: argparse.Namespace,
    console: Console,
    prompt_reader: PromptReader,
    *,
    approval_prompt_reader: PromptReader | None = None,
) -> int:
    try:
        runtime = _build_runtime(
            arguments,
            console,
            approval_prompt_reader=approval_prompt_reader,
        )
    except (ConfigurationError, PathPolicyError, SessionError, OSError, ValueError) as error:
        console.print(f"[bold red]Configuration error:[/bold red] {error}")
        return 2

    _print_runtime(console, arguments.workspace, runtime)
    console.print("Interactive mode. Type [bold]/help[/bold] for commands.")
    initial_task = " ".join(arguments.initial_task).strip()
    try:
        exit_code = _chat_loop(
            runtime.controller,
            runtime.memory,
            console,
            prompt_reader,
            initial_task=initial_task or None,
            memory_store=runtime.store,
            approval_engine=runtime.registry.approval_engine,
            show_history_on_start=runtime.loaded_existing_session,
        )
        runtime.memory.status = SessionStatus.PAUSED.value
        runtime.store.save_state(runtime.memory, status=SessionStatus.PAUSED.value)
        return exit_code
    except OSError as error:
        runtime.memory.status = SessionStatus.FAILED.value
        runtime.store.save_state(runtime.memory, status=SessionStatus.FAILED.value)
        console.print(f"[bold red]Runtime error:[/bold red] {error}")
        return 1
    finally:
        runtime.close()


def _session_command(
    arguments: argparse.Namespace,
    console: Console,
    *,
    prompt_reader: PromptReader | None,
) -> int:
    manager = FileSessionStore()
    try:
        command = arguments.session_command
        if command == "list":
            return _session_list(arguments, console, manager)
        if command == "show":
            _print_session_details(console, manager, manager.load(arguments.session_id))
            return 0
        if command == "current":
            metadata = manager.get_active(arguments.workspace)
            if metadata is None:
                console.print(f"No current session for {arguments.workspace.resolve()}.")
                return 1
            _print_session_summary(console, manager, metadata, current=True)
            return 0
        if command == "use":
            metadata = manager.read_metadata(arguments.session_id)
            workspace = arguments.workspace or Path(metadata.workspace)
            selected = manager.set_active(workspace, metadata.session_id)
            _print_session_summary(console, manager, selected, current=True)
            _print_session_resume_hint(console, selected)
            return 0
        if command == "resume":
            return _session_resume(arguments, console, manager, prompt_reader)
        if command == "new":
            loaded = manager.create(
                workspace=arguments.workspace,
                task=arguments.task or "",
                name=arguments.name,
            )
            _print_session_summary(console, manager, loaded.metadata, current=True)
            return 0
        if command == "rename":
            metadata = manager.rename(arguments.session_id, arguments.name)
            message = Text(f"Renamed session {metadata.short_id} to ")
            message.append(_terminal_text(metadata.name))
            message.append(".")
            console.print(message)
            return 0
        if command == "fork":
            loaded = manager.fork(
                arguments.session_id,
                name=arguments.name,
                set_active=arguments.use,
            )
            _print_session_summary(console, manager, loaded.metadata, current=arguments.use)
            return 0
        if command == "archive":
            metadata = manager.archive(arguments.session_id)
            console.print(f"Archived session {metadata.short_id}.")
            return 0
        if command == "delete":
            if not arguments.yes:
                console.print("Refusing permanent removal without --yes; use archive by default.")
                return 2
            destination = manager.delete(arguments.session_id)
            message = Text("Session moved to recoverable trash: ")
            message.append(_terminal_text(str(destination)))
            console.print(message)
            return 0
        if command == "repair":
            repairs = manager.repair(arguments.session_id)
            if repairs:
                console.print("Session repaired:\n" + "\n".join(f"- {item}" for item in repairs))
            else:
                console.print("Session is consistent; no repairs were needed.")
            return 0
    except (SessionError, OSError, ValueError) as error:
        console.print(f"[bold red]Session error:[/bold red] {error}")
        return 2
    console.print("[red]Unknown session command.[/red]")
    return 2


def _session_list(
    arguments: argparse.Namespace,
    console: Console,
    manager: FileSessionStore,
) -> int:
    if arguments.limit <= 0:
        raise ValueError("--limit must be positive")
    status = SessionStatus(arguments.status) if arguments.status else None
    sessions = manager.list_sessions(
        workspace=arguments.workspace,
        status=status,
        include_archived=arguments.include_archived,
        limit=arguments.limit,
    )
    if not sessions:
        console.print("No sessions found.")
        return 0
    active_by_workspace: dict[str, str | None] = {}
    table = Table(title="RepoRivet sessions")
    table.add_column("")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Step", justify="right")
    table.add_column("Updated")
    table.add_column("Workspace")
    table.add_column("Task")
    table.caption = "* selected session for that workspace"
    for metadata in sessions:
        if metadata.workspace not in active_by_workspace:
            active = manager.get_active(metadata.workspace)
            active_by_workspace[metadata.workspace] = active.session_id if active else None
        status_label = "interrupted" if manager.is_interrupted(metadata) else metadata.status.value
        table.add_row(
            "*" if active_by_workspace[metadata.workspace] == metadata.session_id else "",
            Text(_terminal_text(metadata.short_id)),
            Text(_terminal_text(metadata.name)),
            status_label,
            str(metadata.step),
            metadata.updated_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            Text(_terminal_text(Path(metadata.workspace).name or metadata.workspace)),
            Text(_terminal_text(metadata.task_preview)),
        )
    console.print(table)
    return 0


def _session_resume(
    arguments: argparse.Namespace,
    console: Console,
    manager: FileSessionStore,
    prompt_reader: PromptReader | None,
) -> int:
    if arguments.session_id:
        metadata = manager.read_metadata(arguments.session_id)
        workspace = arguments.workspace or Path(metadata.workspace)
    else:
        workspace = (arguments.workspace or Path.cwd()).resolve()
        metadata = manager.get_active(workspace)
        if metadata is None:
            raise SessionError(
                f"No selected session for workspace {workspace}. Active sessions are "
                "workspace-scoped. Run `reporivet session list`, then resume by ID, or pass "
                "`--workspace <path>`."
            )
    manager.ensure_resumable(metadata)
    runtime_arguments = argparse.Namespace(
        command="chat",
        initial_task=[],
        workspace=Path(workspace),
        config=arguments.config,
        max_steps=arguments.max_steps,
        max_seconds=arguments.max_seconds,
        approval_mode=arguments.approval_mode,
        reasoning=arguments.reasoning,
        session=metadata.session_id,
    )
    reader = prompt_reader or (lambda prompt: Prompt.ask(prompt, console=console))
    return _chat_agent(
        runtime_arguments,
        console,
        reader,
        approval_prompt_reader=prompt_reader,
    )


def _print_session_summary(
    console: Console,
    manager: FileSessionStore,
    metadata: SessionMetadata,
    *,
    current: bool,
) -> None:
    status = "interrupted" if manager.is_interrupted(metadata) else metadata.status.value
    lines = Text()
    lines.append(f"ID:        {_terminal_text(metadata.session_id)}\n")
    lines.append(f"Name:      {_terminal_text(metadata.name)}\n")
    lines.append(f"Workspace: {_terminal_text(metadata.workspace)}\n")
    lines.append(f"Status:    {status}")
    console.print(Panel(lines, title="Current session" if current else "Session"))


def _print_session_resume_hint(console: Console, metadata: SessionMetadata) -> None:
    current_workspace = Path.cwd().resolve()
    session_workspace = Path(metadata.workspace).resolve()
    if current_workspace == session_workspace:
        console.print("Run `reporivet session resume` to continue.")
        return

    hint = Text()
    hint.append("The selection is scoped to the session workspace.\n")
    hint.append(f"Current directory:  {_terminal_text(str(current_workspace))}\n")
    hint.append(f"Session workspace: {_terminal_text(str(session_workspace))}\n")
    hint.append("Resume from any directory: ")
    hint.append(f"reporivet session resume {metadata.short_id}", style="bold")
    console.print(hint)


def _print_session_details(
    console: Console,
    manager: FileSessionStore,
    loaded: LoadedSession,
) -> None:
    metadata = loaded.metadata
    memory = loaded.memory
    status = "interrupted" if manager.is_interrupted(metadata) else metadata.status.value
    details = Text()
    details.append(f"ID:        {_terminal_text(metadata.session_id)}\n")
    details.append(f"Name:      {_terminal_text(metadata.name)}\n")
    details.append(f"Status:    {status}\n")
    details.append(f"Workspace: {_terminal_text(metadata.workspace)}\n")
    details.append(f"Created:   {metadata.created_at.astimezone().isoformat()}\n")
    details.append(f"Updated:   {metadata.updated_at.astimezone().isoformat()}\n")
    details.append(f"Model:     {_terminal_text(metadata.model or 'unknown')}\n")
    details.append(f"Step:      {metadata.step}\n")
    details.append(f"Parent:    {_terminal_text(metadata.parent_session_id or 'none')}\n")
    details.append(f"Task:      {_terminal_text(metadata.task_preview or 'none')}\n")
    modified = ", ".join(sorted(memory.modified_files)) or "none"
    details.append(f"Modified:  {_terminal_text(modified)}\n")
    details.append(f"Verification: {_terminal_text(memory.summary.verification_status)}")
    if memory.reasoning_events:
        decision = memory.reasoning_events[-1]
        details.append(f"\nLast {decision.phase.value}: {_terminal_text(decision.summary)}")
        if decision.next_action is not None:
            details.append(
                f"\nNext action: {_terminal_text(decision.next_action.tool_name)} "
                f"{_terminal_text(decision.next_action.argument_summary)}"
            )
    if memory.observation_events:
        observation = memory.observation_events[-1]
        details.append(f"\nLast observation: {_terminal_text(observation.result_summary)}")
    console.print(Panel(details, title="Session details"))


def _terminal_text(value: str) -> str:
    return escape_terminal_controls(value)


def _build_runtime(
    arguments: argparse.Namespace,
    console: Console,
    *,
    approval_prompt_reader: PromptReader | None = None,
) -> Runtime:
    termination = TerminationConfig(
        max_steps=arguments.max_steps,
        max_seconds=arguments.max_seconds,
    )
    config = load_config(arguments.config)
    memory_config = _memory_config(
        config,
    )
    token_budget_config = _token_budget_config(memory_config)
    estimator = create_token_estimator(
        model=config.api.model,
        tokenizer_encoding=config.api.tokenizer_encoding,
    )
    secrets = (config.api.api_key.get_secret_value(),)
    session_manager = FileSessionStore(secrets=secrets)
    configured_reasoning_mode = ReasoningDisplayMode(
        arguments.reasoning or config.reasoning.display
    )
    reasoning_mode = (
        configured_reasoning_mode if config.reasoning.enabled else ReasoningDisplayMode.OFF
    )
    reasoning_config = config.reasoning.model_copy(update={"display": reasoning_mode})
    reasoning_manager = ReasoningManager(reasoning_config, secrets=secrets)
    calibration_store = TokenCalibrationStore(session_manager.root / "token-calibration.json")
    token_manager = TokenBudgetManager(
        estimator=estimator,
        config=token_budget_config,
        calibration_store=calibration_store,
        base_url=str(config.api.base_url),
        model=config.api.model,
    )
    workspace = arguments.workspace.expanduser().resolve()
    reference = getattr(arguments, "session", None)
    if reference is None:
        active = session_manager.get_active(workspace)
        reference = active.session_id if active is not None else None
    loaded_existing_session = reference is not None
    if reference is None:
        task_preview = _argument_task_preview(arguments)
        loaded = session_manager.create(
            workspace=workspace,
            task=task_preview,
            model=config.api.model,
        )
    else:
        loaded = session_manager.load(reference)
        session_manager.ensure_resumable(loaded.metadata)
        if loaded.metadata.workspace != str(workspace):
            raise SessionError(
                f"Session {loaded.metadata.session_id} belongs to {loaded.metadata.workspace}, "
                f"not {workspace}"
            )
        session_manager.set_active(workspace, loaded.metadata.session_id)

    session_lock = session_manager.lock(loaded.metadata.session_id)
    session_lock.__enter__()
    try:
        store = loaded.store
        memory = loaded.memory
        memory.config = memory_config
        changed_files = store.validate_workspace(memory, workspace)
        if changed_files:
            store.log("external_files_changed", paths=changed_files)
        unfinished_calls = store.reconcile_interrupted_tool_calls(memory)
        if unfinished_calls:
            store.log("interrupted_tool_calls_detected", calls=unfinished_calls)
            store.save_state(memory, status=SessionStatus.PAUSED.value)
    except Exception:
        session_lock.__exit__(None, None, None)
        raise
    try:
        runtime_events = CompositeEventSink(
            store,
            ConsoleEventReporter(
                console,
                secrets=secrets,
                reasoning_mode=reasoning_mode,
            ),
        )
        approval_engine = _build_approval_engine(
            config=config,
            arguments=arguments,
            memory=memory,
            event_logger=runtime_events,
            console=console,
            prompt_reader=approval_prompt_reader,
        )
        registry = create_default_registry(
            workspace,
            approval_engine=approval_engine,
            snapshot_dir=store.session_dir / "snapshots",
            event_logger=runtime_events,
            initial_workspace_revision=memory.workspace_revision,
        )
        controller = AgentController(
            model_client=OpenAICompatibleClient(config.api),
            tool_registry=registry,
            context_manager=ContextManager(token_manager=token_manager),
            termination_policy=TerminationPolicy(termination),
            event_logger=runtime_events,
            memory_store=store,
            reasoning_manager=reasoning_manager,
        )
        memory.status = SessionStatus.RUNNING.value
        store.save_state(memory, status=SessionStatus.RUNNING.value)
        store.log("session_runtime_started", command=arguments.command)
    except Exception:
        session_lock.__exit__(None, None, None)
        raise
    return Runtime(
        config=config,
        registry=registry,
        store=store,
        memory=memory,
        controller=controller,
        session_manager=session_manager,
        session_lock=session_lock,
        loaded_existing_session=loaded_existing_session,
    )


def _argument_task_preview(arguments: argparse.Namespace) -> str:
    if arguments.command == "run":
        return " ".join(arguments.task).strip()
    if arguments.command == "chat":
        return " ".join(arguments.initial_task).strip()
    return ""


def _build_approval_engine(
    *,
    config: AppConfig,
    arguments: argparse.Namespace,
    memory: MemoryState,
    event_logger: EventSink,
    console: Console,
    prompt_reader: PromptReader | None,
) -> ApprovalEngine:
    mode = ApprovalMode(
        arguments.approval_mode or memory.approval_mode_override or config.approval.mode
    )
    interactive = arguments.command == "chat" or prompt_reader is not None or sys.stdin.isatty()
    human_approver = (
        TerminalHumanApprover(
            console,
            reader=prompt_reader,
            timeout_seconds=config.approval.approval_timeout_seconds,
        )
        if interactive
        else NonInteractiveHumanApprover(config.approval.non_interactive)
    )
    llm_reviewer = None
    if mode == ApprovalMode.LLM_AUTO and config.approval.llm.enabled:
        llm_reviewer = OpenAIApprovalReviewer(
            config.api,
            model=config.approval.llm.model,
            timeout_seconds=config.approval.llm.timeout_seconds,
        )
    safety = config.approval.safety
    risk_levels = {
        "safe": RiskLevel.SAFE,
        "low": RiskLevel.LOW,
        "medium": RiskLevel.MEDIUM,
    }
    engine = ApprovalEngine(
        mode=mode,
        normalizer=RequestNormalizer(arguments.workspace),
        risk_analyzer=RiskAnalyzer(
            ApprovalFactAnalyzer(
                trusted_executable_directories=config.approval.toolchains.trusted_directories
            )
        ),
        hard_policy=HardSafetyPolicy(
            HardSafetySettings(
                deny_outside_workspace_write=safety.deny_outside_workspace_write,
                deny_privilege_escalation=safety.deny_privilege_escalation,
                deny_secret_access=safety.deny_secret_access,
                deny_device_access=safety.deny_device_access,
            )
        ),
        grant_store=ApprovalGrantStore(
            memory,
            remember_approvals=config.approval.remember_session_approvals,
            remember_denials=config.approval.remember_session_denials,
        ),
        human_approver=human_approver,
        llm_reviewer=llm_reviewer,
        max_llm_risk=risk_levels[config.approval.llm.max_auto_approve_risk],
        event_logger=event_logger,
    )
    if arguments.approval_mode is not None:
        memory.approval_mode_override = mode
    return engine


def _memory_config(config: AppConfig) -> MemoryConfig:
    return MemoryConfig(
        max_context_tokens=config.api.context_window_tokens,
        active_prompt_limit=config.token.active_prompt_limit,
        reserved_output_tokens=config.token.reserved_output_tokens,
        reserved_tool_result_tokens=config.token.reserved_tool_result_tokens,
        safety_margin_ratio=config.token.safety_margin_ratio,
        compaction_threshold=config.token.soft_limit_ratio,
        hard_limit_threshold=config.token.hard_limit_ratio,
        default_correction_factor=config.token.default_correction_factor,
        calibration_window=config.token.calibration_window,
        max_context_overflow_retries=config.token.max_context_overflow_retries,
    )


def _token_budget_config(memory_config: MemoryConfig) -> TokenBudgetConfig:
    return TokenBudgetConfig(
        context_limit=memory_config.max_context_tokens,
        active_prompt_limit=memory_config.active_prompt_limit,
        reserved_output_tokens=memory_config.reserved_output_tokens,
        reserved_tool_result_tokens=memory_config.reserved_tool_result_tokens,
        safety_margin_ratio=memory_config.safety_margin_ratio,
        soft_limit_ratio=memory_config.compaction_threshold,
        hard_limit_ratio=memory_config.hard_limit_threshold,
        default_correction_factor=memory_config.default_correction_factor,
        calibration_window=memory_config.calibration_window,
        max_context_overflow_retries=memory_config.max_context_overflow_retries,
    )


def _print_runtime(console: Console, workspace: Path, runtime: Runtime) -> None:
    console.print(
        Panel.fit(
            f"Workspace: {workspace.resolve()}\n"
            f"Model: {runtime.config.api.model}\n"
            f"Context window: {runtime.config.api.context_window_tokens} tokens\n"
            f"Active prompt limit: {runtime.config.token.active_prompt_limit} tokens\n"
            f"Reserved output headroom: {runtime.config.token.reserved_output_tokens} tokens\n"
            f"Token estimator: {runtime.controller.context_manager.token_manager.name}\n"
            f"Approval mode: {runtime.registry.approval_engine.mode.value}\n"
            f"Decision trace: {runtime.controller.reasoning_manager.config.display.value}\n"
            f"Safe prompt budget: "
            f"{runtime.controller.context_manager.token_manager.config.prompt_budget} tokens\n"
            f"Session: {runtime.store.session_dir}",
            title="RepoRivet",
        )
    )


def _chat_loop(
    agent: ConversationAgent,
    memory: MemoryState,
    console: Console,
    prompt_reader: PromptReader,
    *,
    initial_task: str | None = None,
    memory_store: MemoryStore | None = None,
    approval_engine: ApprovalModeManager | None = None,
    show_history_on_start: bool = False,
) -> int:
    if show_history_on_start:
        _print_chat_history(memory, console, heading=True)

    pending_task = initial_task

    while True:
        if pending_task is not None:
            user_input = pending_task
            pending_task = None
            console.print(f"[bold cyan]You:[/bold cyan] {user_input}")
        else:
            try:
                user_input = prompt_reader("[bold cyan]You[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\nConversation ended.")
                return 0

        request = user_input.strip()
        if not request:
            continue
        if request.startswith("/"):
            should_exit, memory_changed = _handle_chat_command(
                request,
                memory,
                console,
                approval_engine=approval_engine,
            )
            if memory_changed and memory_store is not None:
                memory_store.save_state(memory, status=memory.status)
            if should_exit:
                return 0
            continue

        result = agent.run(request, memory=memory)
        _print_result(console, result)
        if memory_store is not None:
            memory.status = SessionStatus.PAUSED.value
            memory_store.save_state(memory, status=SessionStatus.PAUSED.value)


def _handle_chat_command(
    command: str,
    memory: MemoryState,
    console: Console,
    *,
    approval_engine: ApprovalModeManager | None = None,
) -> tuple[bool, bool]:
    normalized = command.lower()
    if normalized in {"/exit", "/quit"}:
        console.print("Conversation ended.")
        return True, False
    if normalized == "/help":
        console.print(
            "/help     Show commands\n"
            "/history  Show remembered conversation\n"
            "/clear    Clear remembered conversation\n"
            "/compact  Compress recent conversation safely\n"
            "/compact aggressive  Keep a smaller recent window\n"
            "/approval  Show the current approval mode\n"
            "/approval <mode>  Switch approval mode immediately\n"
            "/exit     End interactive mode"
        )
        return False, False
    if normalized == "/history":
        _print_chat_history(memory, console)
        return False, False
    if normalized == "/clear":
        memory.clear_recent_conversation()
        console.print("Recent conversation cleared; fixed task and structured state preserved.")
        return False, True
    if normalized in {"/compact", "/compact aggressive"}:
        before_messages = len(memory.messages)
        before_characters = sum(len(message.content or "") for message in memory.messages)
        recovery_level = 2 if normalized.endswith(" aggressive") else 1
        removed = ConversationCompactor().compact(
            memory,
            aggressive=True,
            recovery_level=recovery_level,
        )
        after_characters = sum(len(message.content or "") for message in memory.messages)
        reduced_characters = before_characters - after_characters
        if removed or reduced_characters:
            console.print(
                "Manual compaction complete: "
                f"removed {removed} messages, reduced {reduced_characters} characters, "
                f"and kept {len(memory.messages)} recent messages. "
                "Fixed task and structured memory were preserved."
            )
            return False, True
        console.print(
            f"No eligible recent history to compact ({before_messages} messages retained)."
        )
        return False, False
    if normalized.startswith("/compact"):
        console.print("Usage: /compact or /compact aggressive")
        return False, False
    if normalized == "/approval":
        if approval_engine is None:
            console.print("Approval controls are unavailable in this runtime.")
            return False, False
        modes = ", ".join(mode.value for mode in ApprovalMode)
        console.print(
            f"Current approval mode: [bold]{approval_engine.mode.value}[/bold]\n"
            f"Available modes: {modes}\n"
            "Usage: /approval <mode>"
        )
        return False, False
    if normalized.startswith("/approval "):
        if approval_engine is None:
            console.print("Approval controls are unavailable in this runtime.")
            return False, False
        parts = normalized.split()
        if len(parts) != 2:
            console.print("Usage: /approval <mode>")
            return False, False
        try:
            mode = ApprovalMode(parts[1])
        except ValueError:
            modes = ", ".join(item.value for item in ApprovalMode)
            console.print(f"Unknown approval mode: {parts[1]}. Available modes: {modes}")
            return False, False
        if mode == approval_engine.mode:
            console.print(f"Approval mode is already {mode.value}.")
            return False, False
        previous = approval_engine.mode
        approval_engine.set_mode(mode)
        console.print(f"Approval mode changed: {previous.value} → {mode.value}")
        return False, True
    console.print(f"Unknown command: {command}. Type /help for commands.")
    return False, False


def _print_chat_history(
    memory: MemoryState,
    console: Console,
    *,
    heading: bool = False,
) -> None:
    """Render the remembered user-visible conversation as literal terminal text."""
    visible_messages = [
        message
        for message in memory.messages
        if message.role in {"user", "assistant"} and message.content
    ]
    has_history = memory.fixed is not None or bool(memory.task_updates) or bool(visible_messages)

    if heading:
        console.print(Text("Conversation history", style="bold"))
    if not has_history:
        console.print("No conversation history.")
        return

    if memory.fixed is not None:
        _print_history_entry(console, "Original task", memory.fixed.original_task)
    for update in memory.task_updates:
        _print_history_entry(console, "Task update", update)
    for message in visible_messages:
        label = "You" if message.role == "user" else "RepoRivet"
        _print_history_entry(console, label, message.content or "")


def _print_history_entry(console: Console, label: str, content: str) -> None:
    line = Text()
    line.append(f"{label}: ", style="bold")
    line.append(_terminal_text(content))
    console.print(line)


def _print_result(console: Console, result: AgentResult) -> None:
    color = {
        "success": "green",
        "incomplete": "yellow",
        "blocked": "yellow",
        "stopped": "yellow",
        "error": "red",
    }[result.status]
    content = Text("Status: ")
    content.append(result.status, style=color)
    content.append(f"\nModel steps: {result.step_count}")
    content.append(f"\nTool calls: {result.tool_call_count}")
    modified_files = (
        ", ".join(escape_terminal_controls(path) for path in result.modified_files) or "none"
    )
    content.append(f"\nModified files: {modified_files}")
    verification = result.verification_status.value.replace("_", " ")
    content.append(f"\nVerification: {verification}")
    if result.reason:
        content.append(f"\nReason: {escape_terminal_controls(result.reason)}")
    content.append("\n\n")
    content.append(escape_terminal_controls(result.summary))
    console.print(Panel(content, title="Result"))


def main() -> None:
    """Console-script entry point."""
    raise SystemExit(cli())
