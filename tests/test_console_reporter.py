from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from repo_rivet.approval.engine import ApprovalEngine
from repo_rivet.approval.grant_store import ApprovalGrantStore
from repo_rivet.approval.hard_policy import HardSafetyPolicy
from repo_rivet.approval.human_approver import NonInteractiveHumanApprover
from repo_rivet.approval.models import ApprovalMode
from repo_rivet.approval.normalizer import RequestNormalizer
from repo_rivet.approval.risk_analyzer import RiskAnalyzer
from repo_rivet.memory.models import MemoryState
from repo_rivet.reasoning.models import ReasoningDisplayMode
from repo_rivet.storage.console_reporter import ConsoleEventReporter
from repo_rivet.storage.event_sink import CompositeEventSink
from repo_rivet.tools.base import ToolCall
from repo_rivet.tools.registry import create_default_registry


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def log(self, event_type: str, **data: object) -> None:
        self.events.append((event_type, data))


class RecordingStatus:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.message = ""

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def update(self, message: str) -> None:
        self.message = message


def test_console_reporter_shows_transient_status_while_model_generates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=240)
    reporter = ConsoleEventReporter(console)
    status = RecordingStatus()
    captured: dict[str, str] = {}

    def create_status(message: str, *, spinner: str) -> RecordingStatus:
        captured["message"] = message
        captured["spinner"] = spinner
        return status

    monkeypatch.setattr(console, "status", create_status)

    reporter.log(
        "model_call",
        step=1,
        reasoning_effort="medium",
        reasoning_policy="adaptive",
        reasoning_effort_ceiling="max",
    )

    assert status.started is True
    assert status.stopped is False
    assert "generating the next action" in captured["message"]
    assert "medium reasoning (adaptive up to max)" in captured["message"]
    assert captured["spinner"] == "dots"

    reporter.log("model_call_finished", step=1)

    assert status.stopped is True
    assert reporter._active_status is None
    assert buffer.getvalue() == ""


def test_console_reporter_updates_model_status_with_safe_reasoning_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=240)
    reporter = ConsoleEventReporter(console)
    status = RecordingStatus()

    monkeypatch.setattr(console, "status", lambda _message, *, spinner: status)

    reporter.log("model_call", step=1)
    reporter.log(
        "model_stream_progress",
        activity_phase="evaluating_options",
        elapsed_seconds=24.2,
        reasoning_chars=10_000,
    )

    assert "analyzing context" in status.message
    assert "evaluating the next action" in status.message
    assert "24s" in status.message
    assert buffer.getvalue() == ""


def test_console_reporter_shows_provider_reasoning_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(file=StringIO(), force_terminal=False, color_system=None)
    reporter = ConsoleEventReporter(console)
    status = RecordingStatus()
    captured: dict[str, str] = {}

    def create_status(message: str, *, spinner: str) -> RecordingStatus:
        captured["message"] = message
        return status

    monkeypatch.setattr(console, "status", create_status)

    reporter.log(
        "model_reasoning_effort_mapped",
        requested_effort="xhigh",
        applied_effort="high",
    )

    assert "high reasoning" in captured["message"]
    assert "provider mapping from xhigh" in captured["message"]


def test_console_reporter_shows_approval_model_review_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=240)
    reporter = ConsoleEventReporter(console)
    status = RecordingStatus()
    captured: dict[str, str] = {}

    def create_status(message: str, *, spinner: str) -> RecordingStatus:
        captured["message"] = message
        captured["spinner"] = spinner
        return status

    monkeypatch.setattr(console, "status", create_status)

    reporter.log("llm_approval_review_started", tool="edit_file")

    assert status.started is True
    assert "Approval model is reviewing edit_file" in captured["message"]
    assert captured["spinner"] == "dots"

    reporter.log("llm_approval_reviewed", tool="edit_file")

    assert status.stopped is True
    assert reporter._active_status is None


def test_console_reporter_shows_adaptive_plan_review_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, width=240)
    reporter = ConsoleEventReporter(console)
    status = RecordingStatus()
    captured: dict[str, str] = {}

    def create_status(message: str, *, spinner: str) -> RecordingStatus:
        captured["message"] = message
        captured["spinner"] = spinner
        return status

    monkeypatch.setattr(console, "status", create_status)

    reporter.log("auto_plan_review_started")

    assert status.started is True
    assert "Evaluating whether Plan Mode is needed" in captured["message"]
    assert captured["spinner"] == "dots"

    reporter.log("auto_plan_reviewed", decision="execute")

    assert status.stopped is True
    assert reporter._active_status is None


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        ("list_files", "Listing workspace files"),
        ("search_text", "Searching workspace text"),
        ("read_file", "Reading file"),
        ("write_file", "Creating file"),
        ("edit_file", "Applying edits"),
        ("run_command", "Running command"),
        ("run_verification", "Running verification"),
        ("git_diff", "Inspecting Git changes"),
        ("git_status", "Inspecting Git status"),
        ("future_tool", "Running tool"),
    ],
)
def test_console_reporter_animates_every_tool_with_a_future_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    message: str,
) -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, color_system=None, width=240)
    reporter = ConsoleEventReporter(console)
    status = RecordingStatus()
    captured: dict[str, str] = {}

    def create_status(value: str, *, spinner: str) -> RecordingStatus:
        captured["message"] = value
        captured["spinner"] = spinner
        return status

    monkeypatch.setattr(console, "status", create_status)

    reporter.log("approved_tool_started", tool=tool)

    assert status.started is True
    assert status.stopped is False
    assert message in captured["message"]
    assert tool in captured["message"]
    assert captured["spinner"] == "dots"

    reporter.log("approved_tool_executed", tool=tool, ok=True)

    assert status.stopped is True
    assert reporter._active_status is None
    assert buffer.getvalue() == ""


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


def test_console_reporter_shows_controller_owned_plan_progress() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )

    reporter.log(
        "plan_step_started",
        plan_id="plan-hidden",
        step_index=1,
        step_count=2,
        title="Update collision handling",
        status="running",
    )
    reporter.log(
        "plan_step_finished",
        plan_id="plan-hidden",
        step_index=1,
        step_count=2,
        title="Update collision handling",
        status="completed",
    )

    assert buffer.getvalue().splitlines() == [
        "[PLAN 1/2] Update collision handling · running",
        "[PLAN 1/2] Update collision handling · completed",
    ]
    assert "plan-hidden" not in buffer.getvalue()


def test_console_reporter_announces_automatic_plan_transition() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )

    reporter.log(
        "auto_plan_started",
        source="controller",
        reason="task description declares project-wide scope",
    )

    assert buffer.getvalue().strip() == (
        "[PLAN] entered read-only planning (controller) · "
        "task description declares project-wide scope"
    )


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
        guidance="Read src/app.py first and change only the failing branch",
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
    assert "direction: Read src/app.py first and change only the failing branch" in output
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
        "… run_command · running command",
        "✓ run_command · exit 0 · 2.5s",
    ]


def test_console_reporter_marks_file_creation_as_started() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )

    reporter.log(
        "approval_decided",
        tool="write_file",
        action="allow",
        source="allow_all_mode",
        risk="medium",
    )
    reporter.log("approved_tool_started", tool="write_file")
    reporter.log(
        "tool_result",
        name="write_file",
        ok=True,
        metadata={"path": "game.cpp", "bytes": 4_096, "line_count": 120},
    )

    assert buffer.getvalue().splitlines() == [
        "… write_file · creating file",
        "✓ write_file",
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
    assert "abc1234567890" not in output


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


def test_console_reporter_shows_structured_summary_trace_without_tool_duplication() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240),
        reasoning_mode=ReasoningDisplayMode.SUMMARY,
    )

    reporter.log("reasoning", phase="plan", summary="Inspect, edit, and verify.")
    reporter.log("action", tool="read_file", argument_summary="src/app.py")
    reporter.log(
        "observation",
        event_id="obs-123",
        ok=True,
        result_summary="Read src/app.py:1-20.",
    )
    reporter.log("tool_result", name="read_file", ok=True)
    reporter.log(
        "observation",
        event_id="obs-verify",
        ok=True,
        result_summary="Command finished with exit code 0.",
    )
    reporter.log(
        "verification_result",
        check_id="tests",
        status="passed",
        reasons=["all registered success criteria passed"],
    )

    assert buffer.getvalue().splitlines() == [
        "[PLAN] Inspect, edit, and verify.",
        "[ACTION] read_file src/app.py",
        "[OBSERVE] Read src/app.py:1-20.",
        "[OBSERVE] Command finished with exit code 0.",
        "[VERIFY] tests passed · all registered success criteria passed",
    ]


def test_final_assessment_is_never_labeled_as_verification() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240),
        reasoning_mode=ReasoningDisplayMode.SUMMARY,
    )

    reporter.log("assessment", summary="The implementation is complete.")

    assert buffer.getvalue().strip() == "[ASSESS] The implementation is complete."


def test_verification_has_one_authoritative_console_result_in_quiet_mode() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )

    reporter.log("tool_result", name="register_verification", ok=True)
    reporter.log("tool_result", name="run_verification", ok=True, metadata={"exit_code": 0})
    reporter.log(
        "verification_result",
        check_id="cpp-build",
        status="passed",
        reasons=["all registered success criteria passed"],
    )

    assert buffer.getvalue().splitlines() == [
        "[VERIFY] cpp-build passed · all registered success criteria passed"
    ]


def test_semantic_auto_approval_names_the_matched_rule() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )

    reporter.log(
        "approval_decided",
        action="allow",
        source="semantic_template:bounded_build",
        tool="run_command",
        risk="medium",
    )

    assert buffer.getvalue().strip() == (
        "✓ run_command · approved by semantic rule (bounded build) · risk medium"
    )


def test_console_reporter_labels_unexecuted_action_as_blocked() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240),
        reasoning_mode=ReasoningDisplayMode.SUMMARY,
    )

    reporter.log(
        "action_blocked",
        tool="run_command",
        reason="A matching decision is required.",
        error_code="decision_validation_failed",
    )
    reporter.log(
        "tool_result",
        name="run_command",
        ok=False,
        error_code="decision_validation_failed",
        executed=False,
    )

    assert buffer.getvalue().splitlines() == [
        "[BLOCKED] run_command · A matching decision is required."
    ]


def test_console_reporter_suppresses_duplicate_blocked_actions_from_one_turn() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240),
        reasoning_mode=ReasoningDisplayMode.SUMMARY,
    )
    payload = {
        "step": 3,
        "tool": "run_command",
        "reason": "The model turn contained multiple state-changing actions.",
        "error_code": "action_cardinality_violation",
    }

    reporter.log("action_blocked", **payload)
    reporter.log("action_blocked", **payload)

    assert buffer.getvalue().splitlines() == [
        "[BLOCKED] run_command · The model turn contained multiple state-changing actions."
    ]


def test_console_reporter_hides_finalization_tool_fallback() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240),
        reasoning_mode=ReasoningDisplayMode.SUMMARY,
    )

    reporter.log(
        "action_blocked",
        step=4,
        tool="run_verification",
        reason="Tools are disabled during finalization.",
        error_code="finalization_tool_disabled",
    )

    assert buffer.getvalue() == ""


def test_console_reporter_trace_is_structured_bounded_and_secret_safe() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240),
        secrets=("opaque-secret",),
        reasoning_mode=ReasoningDisplayMode.TRACE,
    )

    reporter.log(
        "reasoning",
        phase="decision",
        current_goal="modify one branch",
        summary="Use obs-1 and reason-2 without opaque-secret",
        evidence_refs=["obs-1"],
        assumptions=["public API remains stable"],
        open_questions=["none"],
        next_action={
            "tool_name": "edit_file",
            "argument_summary": "src/app.py",
            "expected_result": "one replacement",
        },
        confidence=0.91,
    )

    output = buffer.getvalue()
    assert "[DECISION]" in output
    assert "Goal: modify one branch" in output
    assert "Evidence:" not in output
    assert "obs-1" not in output
    assert "reason-2" not in output
    assert "Next: edit_file src/app.py" in output
    assert "Confidence: 0.91" in output
    assert "opaque-secret" not in output
    assert "[REDACTED]" in output
