"""IncidentLens command-line entry point."""

import argparse
import json

from incidentlens.parser import load_incidents
from incidentlens.summary import build_summary, render_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="incidentlens")
    parser.add_argument("path", help="Path to a pipe-delimited incident log")
    parser.add_argument("--slow-threshold-ms", type=int, default=1_000)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    incidents = load_incidents(arguments.path)
    summary = build_summary(
        incidents,
        slow_threshold_ms=arguments.slow_threshold_ms,
    )
    if arguments.format == "json":
        print(json.dumps(render_markdown(summary)), end="")
    else:
        print(render_markdown(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
