import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "incidents.log"


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "incidentlens.cli", str(FIXTURE), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_markdown_is_the_default_format() -> None:
    result = run_cli()

    assert result.returncode == 0
    assert result.stdout.startswith("# Incident Summary")
    assert "Slow services: payments, search" in result.stdout


def test_json_format_is_machine_readable() -> None:
    result = run_cli("--format", "json")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "by_level": {"ERROR": 2, "INFO": 1, "WARN": 1},
        "slow_services": ["payments", "search"],
        "total_events": 4,
    }
