"""Small event kernel that sequences and validates pure state transitions."""

from __future__ import annotations

from typing import Any

from repo_rivet.agent.invariants import assert_state_invariants
from repo_rivet.agent.runtime import AgentRuntimeState
from repo_rivet.events.models import DomainEvent, DomainEventKind
from repo_rivet.events.reducer import reduce


class RuntimeKernel:
    def __init__(self, state: AgentRuntimeState) -> None:
        self.state = state
        assert_state_invariants(state)

    def dispatch(
        self,
        kind: DomainEventKind,
        *,
        correlation_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> DomainEvent:
        event = DomainEvent(
            seq=self.state.last_event_seq + 1,
            kind=kind,
            correlation_id=correlation_id,
            payload=payload or {},
        )
        next_state = reduce(self.state, event)
        assert_state_invariants(next_state)
        self.state = next_state
        return event
