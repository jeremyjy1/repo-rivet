from io import StringIO
from pathlib import Path

from rich.console import Console

from repo_rivet.agent.controller import AgentResult
from repo_rivet.approval.models import ApprovalMode
from repo_rivet.cli import _chat_loop, _print_result, build_parser, cli
from repo_rivet.memory.context_manager import SYSTEM_PROMPT
from repo_rivet.memory.models import MemoryConfig, MemoryState, Message
from repo_rivet.memory.store import MemoryStore
from repo_rivet.planning.models import WorkflowMode
from repo_rivet.verification.models import VerificationOutcome


class FakeConversationAgent:
    def __init__(self) -> None:
        self.requests: list[tuple[str, list[dict[str, object]]]] = []
        self.workflow_modes: list[WorkflowMode | None] = []

    def run(
        self,
        task: str,
        *,
        memory: MemoryState | None = None,
        workflow_mode: WorkflowMode | None = None,
    ) -> AgentResult:
        assert memory is not None
        self.workflow_modes.append(workflow_mode)
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
            verification_status=VerificationOutcome.NOT_APPLICABLE,
        )


class FakeApprovalEngine:
    def __init__(self) -> None:
        self.mode = ApprovalMode.SAFE_AUTO

    def set_mode(self, mode: ApprovalMode) -> None:
        self.mode = mode


class FakePlanRuntime:
    def __init__(self) -> None:
        self.memory: MemoryState | None = None
        self.approved = False
        self.cancelled = False

    def bind(self, memory: MemoryState) -> None:
        self.memory = memory

    def approve(self) -> object:
        assert self.memory is not None
        self.approved = True
        self.memory.workflow_mode = WorkflowMode.EXECUTE
        return object()

    def cancel(self) -> None:
        assert self.memory is not None
        self.cancelled = True
        self.memory.workflow_mode = WorkflowMode.EXECUTE


def test_result_summary_is_rendered_as_literal_plain_terminal_text() -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    result = AgentResult(
        status="success",
        summary="\x1b[2J[bold]literal model text[/bold]",
        reason=None,
        modified_files=(),
        step_count=1,
        tool_call_count=0,
        verification_status=VerificationOutcome.PASSED,
    )

    _print_result(console, result)

    output = buffer.getvalue()
    assert "[bold]literal model text[/bold]" in output
    assert "\\x1b[2J" in output
    assert "\x1b" not in output
    assert "Verification: passed" in output
    assert "Verification passed:" not in output


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


def test_chat_parser_accepts_explicit_session_id(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        ["chat", "--workspace", str(tmp_path), "--session", "20260828-153022-a7c4e1"]
    )

    assert arguments.session == "20260828-153022-a7c4e1"


def test_parser_exposes_session_management_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["session", "list", "--status", "paused"]).session_command == "list"
    assert parser.parse_args(["session", "show", "a7c4e1"]).session_id == "a7c4e1"
    assert parser.parse_args(["session", "current"]).session_command == "current"
    assert parser.parse_args(["session", "use", "a7c4e1"]).session_command == "use"
    assert parser.parse_args(["session", "resume"]).session_id is None
    assert parser.parse_args(["session", "new", "task"]).task == "task"
    assert parser.parse_args(["session", "rename", "a7c4e1", "name"]).name == "name"
    assert parser.parse_args(["session", "fork", "a7c4e1", "--use"]).use
    assert parser.parse_args(["session", "archive", "a7c4e1"]).session_command == "archive"
    assert parser.parse_args(["session", "delete", "a7c4e1", "--yes"]).yes
    assert parser.parse_args(["session", "repair", "a7c4e1"]).session_command == "repair"


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


def test_parser_exposes_plan_workflow(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        ["plan", "--workspace", str(tmp_path), "inspect", "the", "change"]
    )

    assert arguments.command == "plan"
    assert arguments.task == ["inspect", "the", "change"]


def test_parser_accepts_structured_reasoning_display_mode(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        ["chat", "--workspace", str(tmp_path), "--reasoning", "trace"]
    )

    assert arguments.reasoning == "trace"


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


def test_chat_loop_displays_loaded_session_history_on_start() -> None:
    agent = FakeConversationAgent()
    memory = MemoryState(session_id="chat-test")
    memory.start_task(
        task="inspect [bold]the project[/bold]",
        workspace="/workspace",
        system_prompt=SYSTEM_PROMPT,
        safety_rules=["stay in workspace"],
        completion_rules=["verify changes"],
        max_steps=30,
    )
    memory.messages.append(Message(role="assistant", content="inspection complete"))
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    exit_code = _chat_loop(
        agent,
        memory,
        console,
        lambda _: "/exit",
        show_history_on_start=True,
    )

    output = buffer.getvalue()
    assert exit_code == 0
    assert agent.requests == []
    assert "Conversation history" in output
    assert "Original task: inspect [bold]the project[/bold]" in output
    assert "RepoRivet: inspection complete" in output


def test_chat_can_show_and_switch_approval_mode() -> None:
    agent = FakeConversationAgent()
    approval = FakeApprovalEngine()
    inputs = iter(
        [
            "/approval",
            "/approval always-ask",
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
    assert approval.mode == ApprovalMode.ALWAYS_ASK
    assert "Current approval mode: safe-auto" in output
    assert "Approval mode changed: safe-auto → always-ask" in output
    assert "Unknown approval mode: unsupported" in output
    assert agent.requests == []


def test_chat_plan_command_enters_controller_planning_workflow() -> None:
    agent = FakeConversationAgent()
    plan_runtime = FakePlanRuntime()
    inputs = iter([":plan inspect app.py", "/exit"])
    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    exit_code = _chat_loop(
        agent,
        MemoryState(session_id="chat-plan"),
        console,
        lambda _: next(inputs),
        plan_runtime=plan_runtime,
    )

    assert exit_code == 0
    assert agent.requests[0][0] == "inspect app.py"
    assert agent.workflow_modes == [WorkflowMode.PLANNING]


def test_approval_plan_shortcut_enters_planning_workflow() -> None:
    agent = FakeConversationAgent()
    plan_runtime = FakePlanRuntime()
    inputs = iter(["/approval plan inspect app.py", "/exit"])
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    exit_code = _chat_loop(
        agent,
        MemoryState(session_id="approval-plan-shortcut"),
        console,
        lambda _: next(inputs),
        plan_runtime=plan_runtime,
    )

    assert exit_code == 0
    assert agent.requests[0][0] == "inspect app.py"
    assert agent.workflow_modes == [WorkflowMode.PLANNING]
    assert "Plan Mode is a planning workflow" in buffer.getvalue()


def test_chat_execute_command_approves_plan_without_approving_tools() -> None:
    agent = FakeConversationAgent()
    plan_runtime = FakePlanRuntime()
    memory = MemoryState(
        session_id="chat-execute",
        workflow_mode=WorkflowMode.PLAN_READY,
    )
    inputs = iter([":execute", "/exit"])
    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    exit_code = _chat_loop(
        agent,
        memory,
        console,
        lambda _: next(inputs),
        plan_runtime=plan_runtime,
    )

    assert exit_code == 0
    assert plan_runtime.approved
    assert agent.requests[0][0] == "Execute the user-approved plan."
    assert agent.workflow_modes == [WorkflowMode.EXECUTE]


def test_chat_requires_explicit_review_action_for_ready_plan() -> None:
    agent = FakeConversationAgent()
    memory = MemoryState(
        session_id="chat-plan-ready",
        workflow_mode=WorkflowMode.PLAN_READY,
    )
    inputs = iter(["make changes", "/exit"])
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    exit_code = _chat_loop(agent, memory, console, lambda _: next(inputs))

    assert exit_code == 0
    assert agent.requests == []
    assert "plan is waiting for review" in buffer.getvalue()


def test_chat_manual_aggressive_compaction_preserves_tool_group_and_saves_state(
    tmp_path: Path,
) -> None:
    store = MemoryStore.create(tmp_path / "sessions")
    memory = MemoryState(
        session_id=store.session_id,
        config=MemoryConfig(recent_message_limit=10),
    )
    memory.context_checkpoint = "checkpoint before manual compaction"
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
    assert restored.context_checkpoint is None
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
