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
from repo_rivet.config import AppConfig, ConfigurationError, load_config
from repo_rivet.context.manager import ContextManager
from repo_rivet.llm.openai_compatible import OpenAICompatibleClient
from repo_rivet.memory.budget_manager import TokenBudgetConfig, TokenBudgetManager
from repo_rivet.memory.compactor import ConversationCompactor
from repo_rivet.memory.models import MemoryConfig, MemoryState
from repo_rivet.memory.store import MemoryStore
from repo_rivet.memory.token_calibrator import TokenCalibrationStore
from repo_rivet.memory.token_estimator import create_token_estimator
from repo_rivet.safety.path_policy import PathPolicyError
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


@dataclass(frozen=True, slots=True)
class Runtime:
    config: AppConfig
    registry: ToolRegistry
    store: MemoryStore
    memory: MemoryState
    controller: AgentController


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
    return parser


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Target workspace (default: current directory)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("reporivet.toml"),
        help="Local API configuration file (default: reporivet.toml)",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(".reporivet/sessions"),
        help="Ignored directory for JSONL session logs",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume memory from an existing session directory",
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
    output.print("[red]Unknown command.[/red]")
    return 2


def _run_agent(arguments: argparse.Namespace, console: Console) -> int:
    task = " ".join(arguments.task).strip()
    if not task:
        console.print("[bold red]Task error:[/bold red] Task must not be empty")
        return 2

    try:
        runtime = _build_runtime(arguments, console)
    except (ConfigurationError, PathPolicyError, OSError, ValueError) as error:
        console.print(f"[bold red]Configuration error:[/bold red] {error}")
        return 2

    _print_runtime(console, arguments.workspace, runtime)
    try:
        result = runtime.controller.run(task, memory=runtime.memory)
    except OSError as error:
        console.print(f"[bold red]Runtime error:[/bold red] {error}")
        return 1
    _print_result(console, result)
    return 0 if result.status == "success" else 1


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
    except (ConfigurationError, PathPolicyError, OSError, ValueError) as error:
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
        )
        runtime.store.save_state(runtime.memory, status=runtime.memory.status)
        return exit_code
    except OSError as error:
        console.print(f"[bold red]Runtime error:[/bold red] {error}")
        return 1


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
    calibration_store = TokenCalibrationStore(arguments.log_dir.parent / "token-calibration.json")
    token_manager = TokenBudgetManager(
        estimator=estimator,
        config=token_budget_config,
        calibration_store=calibration_store,
        base_url=str(config.api.base_url),
        model=config.api.model,
    )
    secrets = (config.api.api_key.get_secret_value(),)
    if arguments.resume:
        store = MemoryStore(arguments.resume, secrets=secrets)
        memory = store.load_state()
        memory.config = memory_config
        changed_files = store.validate_workspace(memory, arguments.workspace)
        if changed_files:
            store.log("external_files_changed", paths=changed_files)
    else:
        store = MemoryStore.create(arguments.log_dir, secrets=secrets)
        memory = MemoryState(session_id=store.session_id, config=memory_config)
    runtime_events = CompositeEventSink(
        store,
        ConsoleEventReporter(console, secrets=secrets),
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
        arguments.workspace,
        approval_engine=approval_engine,
    )
    controller = AgentController(
        model_client=OpenAICompatibleClient(config.api),
        tool_registry=registry,
        context_manager=ContextManager(token_manager=token_manager),
        termination_policy=TerminationPolicy(termination),
        event_logger=runtime_events,
        memory_store=store,
    )
    return Runtime(
        config=config,
        registry=registry,
        store=store,
        memory=memory,
        controller=controller,
    )


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
        risk_analyzer=RiskAnalyzer(),
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
        minimum_llm_confidence=config.approval.llm.minimum_confidence,
        max_llm_risk=risk_levels[config.approval.llm.max_auto_approve_risk],
        event_logger=event_logger,
    )
    if arguments.approval_mode is not None:
        memory.approval_mode_override = mode
    return engine


def _memory_config(config: AppConfig) -> MemoryConfig:
    return MemoryConfig(
        max_context_tokens=config.api.context_window_tokens,
        reserved_output_tokens=config.api.max_output_tokens,
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
            f"Maximum output: {runtime.config.api.max_output_tokens} tokens\n"
            f"Token estimator: {runtime.controller.context_manager.token_manager.name}\n"
            f"Approval mode: {runtime.registry.approval_engine.mode.value}\n"
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
) -> int:
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
        if memory.fixed is None and not memory.messages:
            console.print("No conversation history.")
        else:
            if memory.fixed is not None:
                console.print(f"[bold]Original task:[/bold] {memory.fixed.original_task}")
            for update in memory.task_updates:
                console.print(f"[bold]Task update:[/bold] {update}")
            for message in memory.messages:
                if message.role not in {"user", "assistant"} or not message.content:
                    continue
                label = "You" if message.role == "user" else "RepoRivet"
                console.print(f"[bold]{label}:[/bold] {message.content}")
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


def _print_result(console: Console, result: AgentResult) -> None:
    color = {"success": "green", "stopped": "yellow", "error": "red"}[result.status]
    content = Text("Status: ")
    content.append(result.status, style=color)
    content.append(f"\nModel steps: {result.step_count}")
    content.append(f"\nTool calls: {result.tool_call_count}")
    modified_files = (
        ", ".join(escape_terminal_controls(path) for path in result.modified_files) or "none"
    )
    content.append(f"\nModified files: {modified_files}")
    content.append(f"\nVerification passed: {result.verification_success}")
    if result.reason:
        content.append(f"\nReason: {escape_terminal_controls(result.reason)}")
    content.append("\n\n")
    content.append(escape_terminal_controls(result.summary))
    console.print(Panel(content, title="Result"))


def main() -> None:
    """Console-script entry point."""
    raise SystemExit(cli())
