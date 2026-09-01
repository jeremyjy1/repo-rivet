"""Build and render deterministic incident summaries."""

from collections import Counter
from collections.abc import Iterable

from incidentlens.models import Incident


def build_summary(
    incidents: Iterable[Incident],
    *,
    slow_threshold_ms: int = 1_000,
) -> dict[str, object]:
    items = list(incidents)
    by_level = dict(sorted(Counter(item.level for item in items).items()))
    slow_services = sorted(
        {item.service for item in items if item.duration_ms >= slow_threshold_ms}
    )
    return {
        "total_events": len(items),
        "by_level": by_level,
        "slow_services": slow_services,
    }


def render_markdown(summary: dict[str, object]) -> str:
    by_level = summary["by_level"]
    slow_services = summary["slow_services"]
    assert isinstance(by_level, dict)
    assert isinstance(slow_services, list)
    level_rows = "\n".join(f"| {level} | {count} |" for level, count in by_level.items())
    services = ", ".join(str(item) for item in slow_services) or "none"
    return (
        "# Incident Summary\n\n"
        f"Total events: {summary['total_events']}\n\n"
        "| Level | Count |\n"
        "| --- | ---: |\n"
        f"{level_rows}\n\n"
        f"Slow services: {services}\n"
    )
