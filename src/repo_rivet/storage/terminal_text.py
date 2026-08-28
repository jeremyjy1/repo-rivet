"""Make untrusted text inert before writing it to an interactive terminal."""

import re

_TERMINAL_CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def escape_terminal_controls(text: str) -> str:
    """Render control bytes visibly while preserving ordinary text and line breaks."""
    return _TERMINAL_CONTROL_PATTERN.sub(
        lambda match: f"\\x{ord(match.group(0)):02x}",
        text,
    )
