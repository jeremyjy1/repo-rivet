import json
from pathlib import Path

import pytest

from repo_rivet.memory.budget_manager import TokenBudgetConfig, TokenBudgetManager
from repo_rivet.memory.token_calibrator import TokenCalibrationStore, UsageCalibrator
from repo_rivet.memory.token_estimator import ApproximateTokenEstimator


def test_prompt_budget_reserves_output_tool_result_and_safety_margin() -> None:
    config = TokenBudgetConfig(
        context_limit=32_000,
        reserved_output_tokens=4_096,
        reserved_tool_result_tokens=2_048,
        safety_margin_ratio=0.15,
    )

    assert config.prompt_budget == 21_056


def test_calibrator_uses_defaults_small_sample_max_and_larger_sample_p90() -> None:
    calibrator = UsageCalibrator(default_factor=1.25)
    assert calibrator.correction_factor() == 1.25

    calibrator.observe(100, 110)
    calibrator.observe(100, 130)
    assert calibrator.correction_factor() == pytest.approx(1.365)

    for actual in (90, 100, 105):
        calibrator.observe(100, actual)
    assert calibrator.correction_factor() >= 1.05
    assert calibrator.correction_factor() == pytest.approx(1.365)

    calibrator.observe_overflow()
    assert calibrator.correction_factor() == pytest.approx(1.5015)


def test_calibration_is_persisted_and_isolated_by_gateway_and_model(tmp_path: Path) -> None:
    store = TokenCalibrationStore(tmp_path / "token-calibration.json")
    first = UsageCalibrator(ratios=[1.1, 1.2])
    second = UsageCalibrator(ratios=[1.5])

    store.save(base_url="https://one.example/v1", model="model-a", calibrator=first)
    store.save(base_url="https://two.example/v1", model="model-a", calibrator=second)

    restored = store.load(
        base_url="https://one.example/v1/",
        model="model-a",
        max_samples=20,
        default_factor=1.25,
    )
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert restored.ratios == [1.1, 1.2]
    assert len(payload["entries"]) == 2


def test_budget_manager_calibrates_effective_request_estimate(tmp_path: Path) -> None:
    manager = TokenBudgetManager(
        estimator=ApproximateTokenEstimator(safety_factor=1.0),
        config=TokenBudgetConfig(context_limit=32_000),
        calibration_store=TokenCalibrationStore(tmp_path / "token-calibration.json"),
        base_url="https://gateway.example/v1",
        model="model-a",
    )
    messages = [{"role": "user", "content": "hello world"}]
    initial = manager.estimate_request(messages, [])

    manager.observe_usage(estimated=initial.raw, actual=initial.raw * 2)
    corrected = manager.estimate_request(messages, [])

    assert corrected.raw == initial.raw
    assert corrected.effective >= initial.raw * 2.05


def test_budget_manager_caches_tool_schema_estimate_in_state(tmp_path: Path) -> None:
    manager = TokenBudgetManager(
        estimator=ApproximateTokenEstimator(),
        config=TokenBudgetConfig(context_limit=32_000),
        calibration_store=TokenCalibrationStore(tmp_path / "token-calibration.json"),
        base_url="https://gateway.example/v1",
        model="model-a",
    )
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    manager.estimate_request([{"role": "user", "content": "task"}], tools)
    first = manager.state.tool_schema_estimate
    manager.estimate_request([{"role": "user", "content": "next"}], tools)

    assert first > 0
    assert manager.state.tool_schema_estimate == first
