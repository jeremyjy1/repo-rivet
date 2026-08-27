"""Secret-aware JSONL event logger for local agent sessions."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "content",
    "new_text",
    "old_text",
    "password",
    "secret",
    "token",
)
_INLINE_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)


class EventLogger:
    """Append structured events while redacting known credentials."""

    def __init__(
        self,
        path: str | Path,
        *,
        secrets: tuple[str, ...] = (),
        max_string_length: int = 4_000,
    ) -> None:
        if max_string_length <= 0:
            raise ValueError("max_string_length must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._secrets = tuple(secret for secret in secrets if secret)
        self._max_string_length = max_string_length

    def log(self, event_type: str, **data: Any) -> None:
        """Append one sanitized event as a UTF-8 JSON line."""
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event_type,
            "data": self._sanitize(data),
        }
        with self.path.open("a", encoding="utf-8") as log_file:
            json.dump(event, log_file, ensure_ascii=False, default=str)
            log_file.write("\n")

    def sanitize(self, value: Any) -> Any:
        """Return a redacted value suitable for local persistence."""
        return self._sanitize(value)

    def _sanitize(self, value: Any, *, key: str = "") -> Any:
        normalized_key = key.lower()
        if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {
                str(item_key): self._sanitize(item, key=str(item_key))
                for item_key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize(item) for item in value]
        if isinstance(value, str):
            sanitized = value
            for secret in self._secrets:
                sanitized = sanitized.replace(secret, "[REDACTED]")
            sanitized = _INLINE_SECRET_PATTERN.sub(r"\1\2[REDACTED]", sanitized)
            if len(sanitized) > self._max_string_length:
                return f"{sanitized[: self._max_string_length]}... [truncated]"
            return sanitized
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)


def create_session_logger(
    directory: str | Path,
    *,
    secrets: tuple[str, ...] = (),
) -> EventLogger:
    """Create a timestamped session log without using secret material in its name."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return EventLogger(Path(directory) / f"session-{timestamp}.jsonl", secrets=secrets)
