from incidentlens.models import Incident
from incidentlens.summary import build_summary, render_markdown


def incident(service: str, level: str, duration_ms: int) -> Incident:
    return Incident(
        timestamp="2026-08-31T10:00:00Z",
        service=service,
        level=level,
        duration_ms=duration_ms,
        message="sample",
    )


def test_levels_are_aggregated_case_insensitively() -> None:
    summary = build_summary(
        [
            incident("api", "ERROR", 100),
            incident("worker", "error", 200),
            incident("api", "Warn", 300),
        ]
    )

    assert summary["by_level"] == {"ERROR": 2, "WARN": 1}


def test_slow_services_are_unique_and_sorted() -> None:
    summary = build_summary(
        [
            incident("worker", "INFO", 1_500),
            incident("api", "INFO", 1_100),
            incident("worker", "ERROR", 2_000),
        ]
    )

    assert summary["slow_services"] == ["api", "worker"]


def test_markdown_report_keeps_stable_sections() -> None:
    report = render_markdown(
        {
            "total_events": 2,
            "by_level": {"ERROR": 1, "INFO": 1},
            "slow_services": ["payments"],
        }
    )

    assert report.startswith("# Incident Summary")
    assert "| ERROR | 1 |" in report
    assert "Slow services: payments" in report
