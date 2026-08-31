"""CLI server launcher with an already-bound socket and fragment bootstrap token."""

from __future__ import annotations

import socket
import threading
import webbrowser
from argparse import Namespace
from pathlib import Path
from typing import cast

import uvicorn
from rich.console import Console

from repo_rivet.approval.models import ApprovalMode
from repo_rivet.llm.base import ReasoningEffort
from repo_rivet.planning.policy import AutoPlanMode
from repo_rivet.reasoning.policy import ReasoningPolicyMode
from repo_rivet.web.app import create_app

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def run_gui(arguments: Namespace, console: Console) -> int:
    host = str(arguments.host)
    if host not in _LOOPBACK_HOSTS and not arguments.unsafe_network:
        console.print(
            "[bold red]GUI error:[/bold red] Non-loopback binding requires --unsafe-network"
        )
        return 2
    workspace = Path(arguments.workspace).expanduser().resolve()
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(arguments.port)))
    listener.listen(128)
    port = int(listener.getsockname()[1])
    display_host = f"[{host}]" if family == socket.AF_INET6 else host
    origin = f"http://{display_host}:{port}"
    try:
        app = create_app(
            workspace=workspace,
            config_path=arguments.config,
            expected_origin=origin,
            max_steps=arguments.max_steps,
            max_seconds=arguments.max_seconds,
            reasoning=arguments.reasoning,
            default_approval_mode=(
                ApprovalMode(arguments.approval_mode) if arguments.approval_mode else None
            ),
            default_auto_plan=(AutoPlanMode(arguments.auto_plan) if arguments.auto_plan else None),
            default_reasoning_policy=(
                ReasoningPolicyMode(arguments.reasoning_policy)
                if arguments.reasoning_policy
                else None
            ),
            default_reasoning_effort=(
                cast(ReasoningEffort, arguments.reasoning_effort)
                if arguments.reasoning_effort
                else None
            ),
            default_skill=arguments.skill,
            no_skills=arguments.no_skills,
        )
    except (OSError, ValueError) as error:
        listener.close()
        console.print(f"[bold red]GUI error:[/bold red] {error}")
        return 2
    token = app.state.reporivet.auth.bootstrap_token
    url = f"{origin}/#bootstrap={token}"
    console.print(f"RepoRivet GUI: [link={url}]{origin}[/link]")
    console.print(f"Workspace: {workspace}")
    console.print("Press Ctrl-C to stop. The bootstrap link can be used only once.")
    if arguments.open:
        threading.Timer(0.2, webbrowser.open, args=(url,)).start()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        server.run(sockets=[listener])
    except KeyboardInterrupt:
        return 0
    return 0
