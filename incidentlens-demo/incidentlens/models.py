"""Domain records for parsed incidents."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Incident:
    timestamp: str
    service: str
    level: str
    duration_ms: int
    message: str
