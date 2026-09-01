"""Parse IncidentLens pipe-delimited log files."""

from pathlib import Path

from incidentlens.models import Incident


def parse_line(line: str) -> Incident:
    parts = line.rstrip("\n").split("|")
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}")
    timestamp, service, level, duration_text, message = parts
    return Incident(
        timestamp=timestamp,
        service=service,
        level=level,
        duration_ms=int(duration_text),
        message=message,
    )


def load_incidents(path: str | Path) -> list[Incident]:
    source = Path(path)
    incidents: list[Incident] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            incidents.append(parse_line(line))
        except ValueError as error:
            raise ValueError(f"{source}:{line_number}: {error}") from error
    return incidents
