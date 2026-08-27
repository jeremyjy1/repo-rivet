"""Argparse and Rich command-line interface for RepoRivet."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from repo_rivet import __version__
from repo_rivet.agent.controller import AgentController, AgentResult
from repo_rivet.agent.termination import TerminationConfig, TerminationPolicy
from repo_rivet.config import ConfigurationError, load_config
from repo_rivet.context.manager import ContextManager
from repo_rivet.llm.openai_compatible import OpenAICompatibleClient
from repo_rivet.safety.path_policy import PathPolicyError
from repo_rivet.storage.event_logger import create_session_logger
from repo_rivet.tools.registry import create_default_registry


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
    run_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Target workspace (default: current directory)",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("reporivet.toml"),
        help="Local API configuration file (default: reporivet.toml)",
    )
    run_parser.add_argument("--max-steps", type=int, default=30)
    run_parser.add_argument("--max-seconds", type=float, default=600)
    run_parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(".reporivet/sessions"),
        help="Ignored directory for JSONL session logs",
    )
    return parser


def cli(argv: Sequence[str] | None = None, *, console: Console | None = None) -> int:
    """Run the CLI and return a process exit code."""
    output = console or Console()
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        return _run_agent(arguments, output)
    output.print("[red]Unknown command.[/red]")
    return 2


def _run_agent(arguments: argparse.Namespace, console: Console) -> int:
    task = " ".join(arguments.task).strip()
    if not task:
        console.print("[bold red]Task error:[/bold red] Task must not be empty")
        return 2

    try:
        termination = TerminationConfig(
            max_steps=arguments.max_steps,
            max_seconds=arguments.max_seconds,
        )
        config = load_config(arguments.config)
        registry = create_default_registry(arguments.workspace)
        logger = create_session_logger(
            arguments.log_dir,
            secrets=(config.api.api_key.get_secret_value(),),
        )
        controller = AgentController(
            model_client=OpenAICompatibleClient(config.api),
            tool_registry=registry,
            context_manager=ContextManager(),
            termination_policy=TerminationPolicy(termination),
            event_logger=logger,
        )
    except (ConfigurationError, PathPolicyError, OSError, ValueError) as error:
        console.print(f"[bold red]Configuration error:[/bold red] {error}")
        return 2

    console.print(
        Panel.fit(
            f"Workspace: {arguments.workspace.resolve()}\n"
            f"Model: {config.api.model}\n"
            f"Session log: {logger.path}",
            title="RepoRivet",
        )
    )
    try:
        result = controller.run(task)
    except OSError as error:
        console.print(f"[bold red]Runtime error:[/bold red] {error}")
        return 1
    _print_result(console, result)
    return 0 if result.status == "success" else 1


def _print_result(console: Console, result: AgentResult) -> None:
    color = {"success": "green", "stopped": "yellow", "error": "red"}[result.status]
    lines = [
        f"Status: [{color}]{result.status}[/{color}]",
        f"Model steps: {result.step_count}",
        f"Tool calls: {result.tool_call_count}",
        f"Modified files: {', '.join(result.modified_files) or 'none'}",
        f"Verification passed: {result.verification_success}",
    ]
    if result.reason:
        lines.append(f"Reason: {result.reason}")
    lines.extend(("", result.summary))
    console.print(Panel("\n".join(lines), title="Result"))


def main() -> None:
    """Console-script entry point."""
    raise SystemExit(cli())
