from io import StringIO

from rich.console import Console

from repo_rivet.storage.console_reporter import ConsoleEventReporter
from repo_rivet.storage.event_sink import CompositeEventSink


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def log(self, event_type: str, **data: object) -> None:
        self.events.append((event_type, data))


def test_console_reporter_shows_tool_and_approval_lifecycle_without_arguments() -> None:
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
    assert "Tool requested · run_command · step 2" in output
    assert (
        "Approval check · run_command · risk MEDIUM · program pytest (1 arg) "
        "· target /workspace" in output
    )
    assert "Approval allowed · run_command · via LLM reviewer · confidence 0.94" in output
    assert "bounded test command" in output
    assert "Tool completed · run_command · exit 0 · 0.25s" in output
    assert "must-not-leak" not in output
    assert "echo" not in output


def test_console_reporter_shows_denial_without_exposing_inline_secret() -> None:
    buffer = StringIO()
    reporter = ConsoleEventReporter(
        Console(file=buffer, force_terminal=False, color_system=None, width=240)
    )

    reporter.log(
        "approval_decided",
        tool="write_file",
        action="deny",
        source="hard_policy",
        scope="once",
        reason="writes outside workspace are prohibited",
    )
    reporter.log(
        "tool_result",
        tool_call_id="call-1",
        name="write_file",
        ok=False,
        error="api_key=top-secret-value",
    )

    output = buffer.getvalue()
    assert "Approval denied · write_file · via hard safety policy" in output
    assert "writes outside workspace are prohibited" in output
    assert "Tool failed · write_file · api_key=[REDACTED]" in output
    assert "top-secret-value" not in output


def test_composite_event_sink_forwards_each_event() -> None:
    first = RecordingSink()
    second = RecordingSink()
    sink = CompositeEventSink(first, second)

    sink.log("tool_call", name="read_file")

    assert first.events == [("tool_call", {"name": "read_file"})]
    assert second.events == first.events
