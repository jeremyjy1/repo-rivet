from io import StringIO
from pathlib import Path

from rich.console import Console

from repo_rivet.cli import build_parser, cli


def test_run_parser_accepts_workspace_config_and_task(tmp_path: Path) -> None:
    config_path = tmp_path / "local.toml"

    arguments = build_parser().parse_args(
        [
            "run",
            "--workspace",
            str(tmp_path),
            "--config",
            str(config_path),
            "fix",
            "the",
            "bug",
        ]
    )

    assert arguments.workspace == tmp_path
    assert arguments.config == config_path
    assert arguments.task == ["fix", "the", "bug"]


def test_cli_reports_missing_config_without_calling_model(tmp_path: Path) -> None:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)

    exit_code = cli(
        ["run", "--workspace", str(tmp_path), "--config", str(tmp_path / "missing.toml"), "task"],
        console=console,
    )

    assert exit_code == 2
    assert "Configuration file not found" in buffer.getvalue()
