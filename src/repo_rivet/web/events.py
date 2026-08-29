"""Persisted event replay plus non-blocking live wakeups for SSE clients."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EVENT_NAMES = {
    "session_created": "session.created",
    "session_runtime_started": "run.started",
    "session_end": "run.finished",
    "tool_call": "tool.requested",
    "tool_result": "tool.finished",
    "approval_requested": "approval.requested",
    "approval_awaiting_human": "approval.awaiting.human",
    "approval_decided": "approval.resolved",
    "llm_approval_review_started": "approval.review.started",
    "llm_approval_reviewed": "approval.review.completed",
    "llm_approval_review_failed": "approval.review.failed",
    "plan_submitted": "plan.submitted",
    "plan_updated": "plan.updated",
    "plan_approved": "plan.approved",
}


@dataclass(frozen=True, slots=True)
class AgentEvent:
    seq: int
    event_id: str
    session_id: str
    type: str
    timestamp: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class EventBroker:
    """Use size-one queues as wakeups; durable JSONL remains the source of truth."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[None]]] = {}

    def subscribe(self, session_id: str) -> asyncio.Queue[None]:
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self._subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[None]) -> None:
        subscribers = self._subscribers.get(session_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(session_id, None)

    def notify(self, session_id: str) -> None:
        for queue in tuple(self._subscribers.get(session_id, ())):
            with suppress(asyncio.QueueFull):
                queue.put_nowait(None)


class BrokerEventSink:
    def __init__(
        self,
        session_id: str,
        broker: EventBroker,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.session_id = session_id
        self.broker = broker
        self.loop = loop

    def log(self, event_type: str, **data: Any) -> None:
        del event_type, data
        self.loop.call_soon_threadsafe(self.broker.notify, self.session_id)


def read_events(path: Path, session_id: str, *, after: int = 0) -> list[AgentEvent]:
    if not path.is_file():
        return []
    events: list[AgentEvent] = []
    for seq, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if seq <= after:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_type = str(raw.get("event", "event"))
        payload = raw.get("data")
        if not isinstance(payload, dict):
            payload = {"value": payload}
        events.append(
            AgentEvent(
                seq=seq,
                event_id=f"{session_id}:{seq}",
                session_id=session_id,
                type=_EVENT_NAMES.get(raw_type, raw_type.replace("_", ".")),
                timestamp=str(raw.get("timestamp", "")),
                payload=payload,
            )
        )
    return events


def read_event_page(
    path: Path,
    session_id: str,
    *,
    before: int | None = None,
    limit: int = 240,
) -> tuple[list[AgentEvent], bool]:
    """Read one reverse-selected page while returning events in chronological order."""
    if not path.is_file():
        return [], False
    lines = path.read_text(encoding="utf-8").splitlines()
    upper = len(lines) if before is None else min(len(lines), max(0, before - 1))
    lower = max(0, upper - limit)
    events: list[AgentEvent] = []
    for index in range(lower, upper):
        try:
            raw = json.loads(lines[index])
        except json.JSONDecodeError:
            continue
        raw_type = str(raw.get("event", "event"))
        payload = raw.get("data")
        if not isinstance(payload, dict):
            payload = {"value": payload}
        seq = index + 1
        events.append(
            AgentEvent(
                seq=seq,
                event_id=f"{session_id}:{seq}",
                session_id=session_id,
                type=_EVENT_NAMES.get(raw_type, raw_type.replace("_", ".")),
                timestamp=str(raw.get("timestamp", "")),
                payload=payload,
            )
        )
    return events, lower > 0


async def event_stream(
    *,
    path: Path,
    session_id: str,
    broker: EventBroker,
    after: int,
) -> AsyncIterator[str]:
    queue = broker.subscribe(session_id)
    cursor = after
    try:
        while True:
            for event in read_events(path, session_id, after=cursor):
                cursor = event.seq
                data = json.dumps(event.as_dict(), ensure_ascii=False, separators=(",", ":"))
                # Keep one stable SSE event channel. The typed domain event remains in
                # data.type, so clients receive new server-side event kinds without an
                # exhaustive addEventListener registry.
                yield f"id: {event.seq}\ndata: {data}\n\n"
            try:
                await asyncio.wait_for(queue.get(), timeout=15)
            except TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        broker.unsubscribe(session_id, queue)
