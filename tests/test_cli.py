from io import StringIO
from pathlib import Path

from rich.console import Console

from repo_rivet.agent.controller import AgentResult
from repo_rivet.cli import _chat_loop, build_parser, cli
from repo_rivet.memory.context_manager import SYSTEM_PROMPT
from repo_rivet.memory.models import MemoryState, Message


class FakeConversationAgent:
    def __init__(self) -> None:
        self.requests: list[tuple[str, list[dict[str, object]]]] = []

    def run(
        self,
        task: str,
        *,
        memory: MemoryState | None = None,
    ) -> AgentResult:
        assert memory is not None
        self.requests.append(
            (task, [message.model_dump(mode="json") for message in memory.messages])
        )
        memory.start_task(
            task=task,
            workspace="/workspace",
            system_prompt=SYSTEM_PROMPT,
            safety_rules=["stay in workspace"],
            completion_rules=["verify changes"],
            max_steps=30,
        )
        memory.messages.append(Message(role="assistant", content=f"completed {task}"))
        return AgentResult(
            status="success",
            summary=f"completed {task}",
            reason=None,
            modified_files=(),
            step_count=1,
            tool_call_count=0,
            verification_success=False,
        )


def test_run_parser_accepts_workspace_config_and_task(tmp_path: Path) -> None:
    config_path = tmp_path / "local.toml"

    arguments = build_parser().parse_args(
        [
            "run",
            "--workspace",
            str(tmp_path),
            "--config",
            str(config_path),
            "fix",
            "the",
            "bug",
        ]
    )

    assert arguments.workspace == tmp_path
    assert arguments.config == config_path
    assert arguments.task == ["fix", "the", "bug"]


def test_cli_reports_missing_config_without_calling_model(tmp_path: Path) -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    exit_code = cli(
        ["run", "--workspace", str(tmp_path), "--config", str(tmp_path / "missing.toml"), "task"],
        console=console,
    )

    assert exit_code == 2
    assert "Configuration file not found" in buffer.getvalue()


def test_chat_parser_accepts_optional_initial_task(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        ["chat", "--workspace", str(tmp_path), "continue", "the", "task"]
    )

    assert arguments.command == "chat"
    assert arguments.initial_task == ["continue", "the", "task"]


def test_chat_parser_accepts_resume_session(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        ["chat", "--workspace", str(tmp_path), "--resume", str(tmp_path / "session")]
    )

    assert arguments.resume == tmp_path / "session"


def test_chat_loop_remembers_turns_and_can_clear_history() -> None:
    agent = FakeConversationAgent()
    inputs = iter(["continue", "/clear", "fresh task", "/exit"])
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    exit_code = _chat_loop(
        agent,
        MemoryState(session_id="chat-test"),
        console,
        lambda _: next(inputs),
        initial_task="first task",
    )

    assert exit_code == 0
    assert [request[0] for request in agent.requests] == ["first task", "continue", "fresh task"]
    assert agent.requests[0][1] == []
    assert agent.requests[1][1][0]["content"] == "first task"
    assert agent.requests[2][1] == []
    assert "Recent conversation cleared" in buffer.getvalue()


def test_chat_loop_history_and_help_commands_do_not_call_agent() -> None:
    agent = FakeConversationAgent()
    inputs = iter(["task", "/history", "/help", "/exit"])
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    _chat_loop(
        agent,
        MemoryState(session_id="chat-test"),
        console,
        lambda _: next(inputs),
    )

    output = buffer.getvalue()
    assert len(agent.requests) == 1
    assert "Original task: task" in output
    assert "RepoRivet: completed task" in output
    assert "/clear" in output
