"""Persistent per-gateway, per-model correction based on server-reported usage."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UsageCalibrator:
    ratios: list[float] = field(default_factory=list)
    max_samples: int = 20
    default_factor: float = 1.25
    overflow_factor: float = 1.0

    def observe(self, estimated_prompt_tokens: int, actual_prompt_tokens: int) -> None:
        if estimated_prompt_tokens <= 0 or actual_prompt_tokens <= 0:
            return
        ratio = actual_prompt_tokens / estimated_prompt_tokens
        self.ratios.append(max(0.5, min(ratio, 3.0)))
        self.ratios = self.ratios[-self.max_samples :]

    def observe_overflow(self) -> None:
        self.overflow_factor = min(2.0, self.correction_factor() * 1.10)

    def correction_factor(self) -> float:
        if not self.ratios:
            sampled_factor = self.default_factor
        elif len(self.ratios) <= 5:
            sampled_factor = max(1.20, max(self.ratios) * 1.05)
        else:
            ordered = sorted(self.ratios)
            index = math.ceil(len(ordered) * 0.90) - 1
            sampled_factor = max(1.05, ordered[max(0, index)] * 1.05)
        return max(sampled_factor, self.overflow_factor)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": len(self.ratios),
            "ratios": self.ratios,
            "overflow_factor": self.overflow_factor,
        }


class TokenCalibrationStore:
    """Atomically persist calibration samples independently from session state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def key(base_url: str, model: str) -> str:
        return f"{base_url.rstrip('/')}|{model}"

    def load(
        self,
        *,
        base_url: str,
        model: str,
        max_samples: int,
        default_factor: float,
    ) -> UsageCalibrator:
        payload = self._read()
        entry = payload.get("entries", {}).get(self.key(base_url, model), {})
        raw_ratios = entry.get("ratios", [])
        ratios = [
            float(value)
            for value in raw_ratios
            if isinstance(value, int | float) and 0.5 <= float(value) <= 3.0
        ][-max_samples:]
        overflow_factor = entry.get("overflow_factor", 1.0)
        if not isinstance(overflow_factor, int | float):
            overflow_factor = 1.0
        return UsageCalibrator(
            ratios=ratios,
            max_samples=max_samples,
            default_factor=default_factor,
            overflow_factor=max(1.0, min(float(overflow_factor), 2.0)),
        )

    def save(self, *, base_url: str, model: str, calibrator: UsageCalibrator) -> None:
        payload = self._read()
        entries = payload.setdefault("entries", {})
        entries[self.key(base_url, model)] = calibrator.as_dict()
        payload["version"] = 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {"version": 1, "entries": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("entries", {}), dict):
            return {"version": 1, "entries": {}}
        return payload
