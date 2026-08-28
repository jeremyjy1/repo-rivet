"""Crash-safe single-file replacement and race-safe file creation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4


def atomic_replace_bytes(target: Path, data: bytes) -> None:
    """Replace one existing file using a same-directory temporary file."""
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.reporivet.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if target.exists():
            shutil.copymode(target, temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_bytes(target: Path, data: bytes) -> None:
    """Create a file atomically without replacing a path created concurrently."""
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.reporivet.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
