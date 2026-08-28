from io import StringIO
from pathlib import Path

from rich.console import Console

from repo_rivet.approval.engine import ApprovalEngine
from repo_rivet.approval.grant_store import ApprovalGrantStore
from repo_rivet.approval.hard_policy import HardSafetyPolicy
from repo_rivet.approval.human_approver import NonInteractiveHumanApprover
from repo_rivet.approval.models import ApprovalMode
from repo_rivet.approval.normalizer import RequestNormalizer
from repo_rivet.approval.risk_analyzer import RiskAnalyzer
from repo_rivet.memory.models import MemoryState
from repo_rivet.storage.console_reporter import ConsoleEventReporter
from repo_rivet.storage.event_sink import CompositeEventSink
from repo_rivet.tools.base import ToolCall
from repo_rivet.tools.registry import create_default_registry


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def log(self, event_type: str, **data: object) -> None:
        self.events.append((event_type, data))


def test_console_reporter_compacts_tool_and_material_approval_events() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240),
        secrets=("must-not-leak",),
    )

    reporter.log(
        "tool_call",
        step=2,
        tool_call_id="call-1234567890",
        name="run_command",
        arguments={"command": "echo must-not-leak", "content": "must-not-leak"},
    )
    reporter.log(
        "approval_requested",
        tool="run_command",
        risk="medium",
        reasons=["executes project code"],
        affected_paths=["/workspace"],
        program="pytest",
        argument_count=1,
    )
    reporter.log(
        "approval_decided",
        tool="run_command",
        action="allow",
        source="llm_reviewer",
        risk="medium",
        scope="once",
        confidence=0.94,
        reason="bounded test command",
    )
    reporter.log(
        "tool_result",
        tool_call_id="call-1234567890",
        name="run_command",
        ok=True,
        metadata={"exit_code": 0, "duration_seconds": 0.25},
    )

    output = buffer.getvalue()
    assert output.splitlines() == [
        "✓ run_command · approved by LLM reviewer · risk medium",
        "✓ run_command · exit 0 · 0.25s",
    ]
    assert "must-not-leak" not in output
    assert "echo" not in output
    assert "call-1234567890" not in output
    assert "bounded test command" not in output


def test_console_reporter_shows_denial_once_without_exposing_inline_secret() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )

    reporter.log(
        "approval_decided",
        tool="write_file",
        action="deny",
        source="hard_policy",
        risk="critical",
        affected_paths=["/outside/secret.txt"],
        scope="once",
        reason="writes outside workspace are prohibited",
    )
    reporter.log(
        "tool_result",
        tool_call_id="call-1",
        name="write_file",
        ok=False,
        error_code="hard_policy_denied",
        error="api_key=top-secret-value",
    )

    output = buffer.getvalue()
    assert output.count("write_file") == 1
    assert "✗ write_file · denied by hard safety policy" in output
    assert "target /outside/secret.txt" in output
    assert "writes outside workspace are prohibited" in output
    assert "top-secret-value" not in output


def test_console_reporter_hides_routine_approval_and_sanitizes_tool_failure() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )

    reporter.log(
        "approval_requested",
        tool="read_file",
        risk="safe",
        affected_paths=["/workspace/example.py"],
    )
    reporter.log(
        "approval_decided",
        tool="read_file",
        action="allow",
        source="safe_rule",
    )
    reporter.log("tool_result", name="read_file", ok=True, metadata={"path": "example.py"})
    reporter.log(
        "tool_result",
        name="run_command",
        ok=False,
        error="api_key=top-secret-value",
    )

    assert buffer.getvalue().splitlines() == [
        "✓ read_file",
        "✗ run_command · failed · api_key=[REDACTED]",
    ]


def test_console_reporter_keeps_progress_for_slow_auto_approved_command() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )

    reporter.log(
        "approval_decided",
        tool="run_command",
        action="allow",
        source="allow_all_mode",
        risk="medium",
    )
    reporter.log("approved_tool_started", tool="run_command")
    reporter.log(
        "tool_result",
        name="run_command",
        ok=True,
        metadata={"exit_code": 0, "duration_seconds": 2.5},
    )

    assert buffer.getvalue().splitlines() == [
        "… run_command · running",
        "✓ run_command · exit 0 · 2.5s",
    ]


def test_console_reporter_makes_dynamic_markup_and_controls_inert() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240),
        secrets=("opaque-secret",),
    )

    reporter.log(
        "approval_decided",
        tool="[bold]write_file[/bold]",
        action="deny",
        source="hard_policy",
        risk="critical",
        fingerprint="abc1234567890",
        reason="blocked opaque-secret \x1b[2J",
    )

    output = buffer.getvalue()
    assert "[bold]write_file[/bold]" in output
    assert "opaque-secret" not in output
    assert "[REDACTED]" in output
    assert "\\x1b[2J" in output
    assert "\x1b" not in output


def test_real_hard_denial_is_rendered_once_with_its_target(tmp_path: Path) -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )
    memory = MemoryState(session_id="console-denial")
    engine = ApprovalEngine(
        mode=ApprovalMode.ALLOW_ALL,
        normalizer=RequestNormalizer(tmp_path),
        risk_analyzer=RiskAnalyzer(),
        hard_policy=HardSafetyPolicy(),
        grant_store=ApprovalGrantStore(memory),
        human_approver=NonInteractiveHumanApprover(),
        event_logger=reporter,
    )
    registry = create_default_registry(tmp_path, approval_engine=engine)

    result = registry.execute(
        ToolCall(
            id="outside-write",
            name="write_file",
            arguments={"path": "../outside.txt", "content": "blocked"},
        )
    )
    reporter.log(
        "tool_result",
        name="write_file",
        ok=result.ok,
        error=result.error,
        error_code=result.error_code,
        metadata=result.metadata,
    )

    output = buffer.getvalue()
    assert not result.ok
    assert output.count("write_file") == 1
    assert "denied by hard safety policy" in output
    assert str(tmp_path.parent / "outside.txt") in output


def test_composite_event_sink_forwards_each_event() -> None:
    first = RecordingSink()
    second = RecordingSink()
    sink = CompositeEventSink(first, second)

    sink.log("tool_call", name="read_file")

    assert first.events == [("tool_call", {"name": "read_file"})]
    assert second.events == first.events
