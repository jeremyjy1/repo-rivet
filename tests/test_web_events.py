import asyncio
import json
import threading
from pathlib import Path

import pytest

from repo_rivet.approval.models import (
    ApprovalAction,
    ApprovalFacts,
    ApprovalRequest,
    ApprovalScope,
    RiskAssessment,
    RiskLevel,
)
from repo_rivet.planning.models import WorkflowMode
from repo_rivet.web.approvals import WebHumanApprover
from repo_rivet.web.events import EventBroker, read_event_page, read_events
from repo_rivet.web.runtime_manager import RuntimeManager


def test_persisted_events_have_stable_order_and_replay_cursor(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        {"timestamp": "2026-01-01T00:00:00Z", "event": "tool_call", "data": {"name": "read_file"}},
        {"timestamp": "2026-01-01T00:00:01Z", "event": "tool_result", "data": {"ok": True}},
        {
            "timestamp": "2026-01-01T00:00:02Z",
            "event": "session_end",
            "data": {"status": "success"},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    all_events = read_events(path, "session-a")
    replay = read_events(path, "session-a", after=1)

    assert [event.seq for event in all_events] == [1, 2, 3]
    assert [event.type for event in all_events] == [
        "tool.requested",
        "tool.finished",
        "run.finished",
    ]
    assert [event.seq for event in replay] == [2, 3]
    assert replay[0].event_id == "session-a:2"


def test_approval_review_events_have_stable_browser_names(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        {"timestamp": "start", "event": "llm_approval_review_started", "data": {}},
        {"timestamp": "done", "event": "llm_approval_reviewed", "data": {}},
        {"timestamp": "failed", "event": "llm_approval_review_failed", "data": {}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert [event.type for event in read_events(path, "session-a")] == [
        "approval.review.started",
        "approval.review.completed",
        "approval.review.failed",
    ]


def test_event_history_is_loaded_in_bounded_pages(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        {"timestamp": f"2026-01-01T00:00:{index:02d}Z", "event": "observation", "data": {}}
        for index in range(300)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    recent, has_more = read_event_page(path, "session-a", before=301, limit=240)
    earlier, earlier_has_more = read_event_page(path, "session-a", before=61, limit=240)

    assert [event.seq for event in recent] == list(range(61, 301))
    assert has_more
    assert [event.seq for event in earlier] == list(range(1, 61))
    assert not earlier_has_more


def _approval_request() -> ApprovalRequest:
    return ApprovalRequest(
        request_id="request-1",
        session_id="session-1",
        tool_name="write_file",
        arguments={"path": "main.py", "content": "print('ok')"},
        normalized_arguments={"path": "main.py", "content": "print('ok')"},
        workspace="/workspace",
        fingerprint="secret-fingerprint",
        assessment=RiskAssessment(level=RiskLevel.MEDIUM, reasons=["writes a file"]),
        facts=ApprovalFacts(write_paths=["main.py"]),
    )


def _high_risk_approval_request() -> ApprovalRequest:
    request = _approval_request()
    request.assessment.level = RiskLevel.HIGH
    return request


def test_web_approval_is_consumed_exactly_once_and_hides_fingerprint() -> None:
    approver = WebHumanApprover()
    result = []
    thread = threading.Thread(target=lambda: result.append(approver.ask(_approval_request())))
    thread.start()
    pending = approver.wait_for_pending(timeout=1)
    assert pending is not None
    snapshot = approver.snapshot()
    assert snapshot is not None
    assert "fingerprint" not in json.dumps(snapshot)
    assert "arguments" not in snapshot
    assert snapshot["details"]
    assert snapshot["preview"] == {
        "kind": "content",
        "title": "New file content",
        "text": "print('ok')",
    }

    decision = approver.resolve(
        request_id="request-1",
        state_version=pending.state_version,
        action="allow_session",
    )
    thread.join(timeout=1)

    assert decision.action == ApprovalAction.ALLOW
    assert decision.scope == ApprovalScope.SESSION_EXACT
    assert result == [decision]
    with pytest.raises(ValueError, match="No approval is pending"):
        approver.resolve(
            request_id="request-1",
            state_version=pending.state_version,
            action="allow_once",
        )


def test_web_approval_emits_ready_event_after_pending_state_is_visible() -> None:
    approver: WebHumanApprover

    class PendingStateSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object], bool]] = []

        def log(self, event_type: str, **data: object) -> None:
            self.events.append((event_type, data, approver.snapshot() is not None))

    sink = PendingStateSink()
    approver = WebHumanApprover(event_logger=sink)
    thread = threading.Thread(target=lambda: approver.ask(_approval_request()))
    thread.start()
    pending = approver.wait_for_pending(timeout=1)

    assert pending is not None
    assert sink.events == [
        (
            "approval_awaiting_human",
            {
                "request_id": "request-1",
                "tool": "write_file",
                "risk": "medium",
                "state_version": 1,
                "llm_review_available": False,
            },
            True,
        )
    ]
    approver.abort_pending()
    thread.join(timeout=1)


def test_web_approval_rejects_repeating_high_risk_grant() -> None:
    approver = WebHumanApprover()
    thread = threading.Thread(target=lambda: approver.ask(_high_risk_approval_request()))
    thread.start()
    pending = approver.wait_for_pending(timeout=1)
    assert pending is not None

    with pytest.raises(ValueError, match="High-risk"):
        approver.resolve(
            request_id="request-1",
            state_version=pending.state_version,
            action="allow_session",
        )
    approver.abort_pending()
    thread.join(timeout=1)


def test_runtime_manager_rejects_a_second_run_while_stopping(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = RuntimeManager(
            workspace=tmp_path,
            config_path=tmp_path / "config.toml",
            broker=EventBroker(),
        )
        release = asyncio.Event()

        async def worker(*args: object, **kwargs: object) -> None:
            del args, kwargs
            await release.wait()

        manager._worker = worker  # type: ignore[method-assign]
        run = await manager.start(
            session_id="session-1",
            task="first",
            mode=WorkflowMode.EXECUTE,
        )
        await manager.stop("session-1")

        with pytest.raises(ValueError, match="active run"):
            await manager.start(
                session_id="session-1",
                task="second",
                mode=WorkflowMode.EXECUTE,
            )
        release.set()
        assert run.task_handle is not None
        await run.task_handle

    asyncio.run(scenario())
