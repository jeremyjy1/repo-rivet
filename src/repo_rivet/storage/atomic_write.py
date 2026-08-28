"""Atomic JSON persistence helpers."""

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def atomic_write_json(target: Path, value: Any) -> None:
    """Write JSON beside its destination and atomically replace the old file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, default=str)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
