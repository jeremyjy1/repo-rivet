from io import StringIO
from pathlib import Path

from rich.console import Console

from repo_rivet.agent.controller import AgentResult
from repo_rivet.approval.models import ApprovalMode
from repo_rivet.cli import _chat_loop, build_parser, cli
from repo_rivet.memory.context_manager import SYSTEM_PROMPT
from repo_rivet.memory.models import MemoryConfig, MemoryState, Message
from repo_rivet.memory.store import MemoryStore


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


class FakeApprovalEngine:
    def __init__(self) -> None:
        self.mode = ApprovalMode.SAFE_AUTO

    def set_mode(self, mode: ApprovalMode) -> None:
        self.mode = mode


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


def test_parser_accepts_approval_mode_override(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        [
            "run",
            "--workspace",
            str(tmp_path),
            "--approval-mode",
            "always-ask",
            "task",
        ]
    )

    assert arguments.approval_mode == "always-ask"


def test_parser_accepts_read_only_approval_mode(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        ["chat", "--workspace", str(tmp_path), "--approval-mode", "read-only"]
    )

    assert arguments.approval_mode == "read-only"


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
    assert "/compact" in output
    assert "/approval" in output


def test_chat_can_show_and_switch_approval_mode() -> None:
    agent = FakeConversationAgent()
    approval = FakeApprovalEngine()
    inputs = iter(
        [
            "/approval",
            "/approval read-only",
            "/approval unsupported",
            "/exit",
        ]
    )
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=180)

    exit_code = _chat_loop(
        agent,
        MemoryState(session_id="chat-test"),
        console,
        lambda _: next(inputs),
        approval_engine=approval,
    )

    output = buffer.getvalue()
    assert exit_code == 0
    assert approval.mode == ApprovalMode.READ_ONLY
    assert "Current approval mode: safe-auto" in output
    assert "Approval mode changed: safe-auto → read-only" in output
    assert "Unknown approval mode: unsupported" in output
    assert agent.requests == []


def test_chat_manual_aggressive_compaction_preserves_tool_group_and_saves_state(
    tmp_path: Path,
) -> None:
    store = MemoryStore.create(tmp_path / "sessions")
    memory = MemoryState(
        session_id=store.session_id,
        config=MemoryConfig(recent_message_limit=10),
    )
    for index in range(20):
        memory.messages.append(Message(role="assistant", content=f"old {index}"))
    memory.messages.extend(
        [
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "run_command", "arguments": "{}"},
                    }
                ],
            ),
            Message(role="tool", tool_call_id="call-1", content="x" * 10_000),
        ]
    )
    inputs = iter(["/compact aggressive", "/exit"])
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    exit_code = _chat_loop(
        FakeConversationAgent(),
        memory,
        console,
        lambda _: next(inputs),
        memory_store=store,
    )
    restored = store.load_state()

    assert exit_code == 0
    assert restored.compaction_count == 1
    assert len(restored.messages) == 2
    assert restored.messages[0].tool_calls
    assert restored.messages[1].tool_call_id == "call-1"
    assert "aggressively truncated" in (restored.messages[1].content or "")
    assert "Manual compaction complete" in buffer.getvalue()


def test_chat_manual_compaction_reduces_recent_window() -> None:
    memory = MemoryState(
        session_id="manual-compact",
        config=MemoryConfig(recent_message_limit=10),
    )
    for index in range(8):
        memory.messages.append(Message(role="assistant", content=f"recent {index}"))
    inputs = iter(["/compact", "/exit"])
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    _chat_loop(
        FakeConversationAgent(),
        memory,
        console,
        lambda _: next(inputs),
    )

    assert len(memory.messages) == 5
    assert memory.messages[0].content == "recent 3"
    assert memory.compaction_count == 1
