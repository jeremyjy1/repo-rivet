"""Secret-aware JSONL event logger for local agent sessions."""

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "content",
        "password",
        "secret",
        "token",
    }
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
        self._write_lock = threading.Lock()

    def log(self, event_type: str, **data: Any) -> None:
        """Append one sanitized event as a UTF-8 JSON line."""
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event_type,
            "data": self._sanitize(data),
        }
        with self._write_lock, self.path.open("a", encoding="utf-8") as log_file:
            json.dump(event, log_file, ensure_ascii=False, default=str)
            log_file.write("\n")

    def sanitize(self, value: Any) -> Any:
        """Return a redacted value suitable for local persistence."""
        return self._sanitize(value)

    def _sanitize(self, value: Any, *, key: str = "") -> Any:
        if _is_sensitive_key(key):
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


def _is_sensitive_key(key: str) -> bool:
    """Redact credential values without hiding ordinary metrics such as token counts."""
    normalized = key.lower().replace("-", "_")
    if normalized in _SENSITIVE_KEYS:
        return True
    return normalized.endswith(("_api_key", "_authorization", "_password", "_secret", "_token"))


def create_session_logger(
    directory: str | Path,
    *,
    secrets: tuple[str, ...] = (),
) -> EventLogger:
    """Create a timestamped session log without using secret material in its name."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return EventLogger(Path(directory) / f"session-{timestamp}.jsonl", secrets=secrets)
