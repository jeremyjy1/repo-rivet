"""Safety reserves, calibrated request estimates, and usage feedback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from repo_rivet.memory.token_calibrator import TokenCalibrationStore, UsageCalibrator
from repo_rivet.memory.token_estimator import CalibratedTokenEstimator, TokenEstimator


@dataclass(frozen=True, slots=True)
class TokenBudgetConfig:
    context_limit: int
    active_prompt_limit: int = 65_536
    reserved_output_tokens: int = 4_096
    reserved_tool_result_tokens: int = 2_048
    safety_margin_ratio: float = 0.15
    soft_limit_ratio: float = 0.70
    hard_limit_ratio: float = 0.85
    default_correction_factor: float = 1.25
    calibration_window: int = 20
    max_context_overflow_retries: int = 2

    @property
    def prompt_budget(self) -> int:
        fixed_reserve = self.reserved_output_tokens + self.reserved_tool_result_tokens
        safety_margin = int(self.context_limit * self.safety_margin_ratio)
        return max(0, self.context_limit - fixed_reserve - safety_margin)

    @property
    def request_budget(self) -> int:
        """Return the cost-aware request ceiling within the provider-safe budget."""
        return min(self.prompt_budget, self.active_prompt_limit)


@dataclass(frozen=True, slots=True)
class RequestTokenEstimate:
    raw: int
    effective: int
    correction_factor: float


@dataclass(slots=True)
class TokenBudgetState:
    tool_schema_estimate: int = 0
    fixed_prompt_estimate: int = 0


class TokenBudgetManager:
    """Own estimation, calibration persistence, and budget thresholds."""

    def __init__(
        self,
        *,
        estimator: TokenEstimator,
        config: TokenBudgetConfig,
        calibration_store: TokenCalibrationStore | None,
        base_url: str,
        model: str,
    ) -> None:
        self.config = config
        self.calibration_store = calibration_store
        self.base_url = base_url
        self.model = model
        self.state = TokenBudgetState()
        calibrator = (
            calibration_store.load(
                base_url=base_url,
                model=model,
                max_samples=config.calibration_window,
                default_factor=config.default_correction_factor,
            )
            if calibration_store is not None
            else UsageCalibrator(
                max_samples=config.calibration_window,
                default_factor=config.default_correction_factor,
            )
        )
        self.estimator = CalibratedTokenEstimator(estimator, calibrator)

    @property
    def name(self) -> str:
        return self.estimator.name

    @property
    def correction_factor(self) -> float:
        return self.estimator.calibrator.correction_factor()

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> RequestTokenEstimate:
        raw = self.estimator.raw_estimate_request(messages, tools)
        self.state.tool_schema_estimate = self.estimator.base.estimate_request([], tools)
        factor = self.correction_factor
        return RequestTokenEstimate(
            raw=raw,
            effective=math.ceil(raw * factor),
            correction_factor=factor,
        )

    def pressure_level(
        self,
        effective_tokens: int,
    ) -> Literal["normal", "compact", "aggressive", "overflow"]:
        budget = self.config.prompt_budget
        if effective_tokens > budget:
            return "overflow"
        if effective_tokens >= int(budget * self.config.hard_limit_ratio):
            return "aggressive"
        compact_at = min(
            int(budget * self.config.soft_limit_ratio),
            self.config.active_prompt_limit,
        )
        if effective_tokens >= compact_at:
            return "compact"
        return "normal"

    def observe_usage(self, *, estimated: int, actual: int) -> None:
        self.estimator.calibrator.observe(estimated, actual)
        self._save()

    def observe_overflow(self) -> None:
        self.estimator.calibrator.observe_overflow()
        self._save()

    def _save(self) -> None:
        if self.calibration_store is not None:
            self.calibration_store.save(
                base_url=self.base_url,
                model=self.model,
                calibrator=self.estimator.calibrator,
            )
