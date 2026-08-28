"""Exclusive, inspectable per-session process locks."""

import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn

from repo_rivet.session.errors import (
    SessionAlreadyRunning,
    SessionCorrupted,
    SessionLockStale,
)


def process_is_alive(pid: int) -> bool:
    """Return whether a local process exists without signalling it."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SessionLock:
    """Hold a lock file until the current runtime reaches a safe boundary."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._owned = False

    def __enter__(self) -> "SessionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            self._raise_existing_lock()
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False)
            output.write("\n")
        self._owned = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False

    def _raise_existing_lock(self) -> NoReturn:
        try:
            payload: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            hostname = str(payload["hostname"])
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise SessionCorrupted(f"Invalid session lock {self.path}: {error}") from None
        if hostname == socket.gethostname() and not process_is_alive(pid):
            raise SessionLockStale(
                f"Session has a stale lock from process {pid}; run `reporivet session repair`"
            )
        raise SessionAlreadyRunning(f"Session is already locked by {hostname} process {pid}")
