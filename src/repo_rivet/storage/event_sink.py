"""Small event-sink composition primitives shared by runtime observers."""

from typing import Any, Protocol


class EventSink(Protocol):
    def log(self, event_type: str, **data: Any) -> None:
        """Consume one structured runtime event."""
        ...


class CompositeEventSink:
    """Synchronously forward each event to persistent and interactive observers."""

    def __init__(self, *sinks: EventSink) -> None:
        self.sinks = sinks

    def log(self, event_type: str, **data: Any) -> None:
        for sink in self.sinks:
            sink.log(event_type, **data)
