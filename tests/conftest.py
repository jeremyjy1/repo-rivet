"""Shared test controls, including explicit opt-in for billable provider checks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from repo_rivet.config import ApiConfig, ConfigurationError, load_config

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("live-api")
    group.addoption(
        "--run-live-api",
        action="store_true",
        default=False,
        help="Run tests that call a real API and may incur provider charges.",
    )
    group.addoption(
        "--live-api-config",
        metavar="PATH",
        default=None,
        help=(
            "RepoRivet TOML used by live API tests "
            "(default: REPORIVET_LIVE_CONFIG or reporivet.toml)."
        ),
    )


@pytest.fixture(scope="session")
def live_api_config(request: pytest.FixtureRequest) -> ApiConfig:
    """Load billable-test credentials only after an explicit cost opt-in."""
    enabled = bool(request.config.getoption("--run-live-api")) or (
        os.environ.get("REPORIVET_RUN_LIVE_API", "").strip().casefold() in _TRUE_VALUES
    )
    if not enabled:
        pytest.skip(
            "real API tests are disabled; pass --run-live-api or set REPORIVET_RUN_LIVE_API=1"
        )

    configured_path = request.config.getoption("--live-api-config") or os.environ.get(
        "REPORIVET_LIVE_CONFIG"
    )
    if configured_path:
        api = _load_api_config(Path(str(configured_path)))
    elif Path("reporivet.toml").is_file():
        api = _load_api_config(Path("reporivet.toml"))
    else:
        api = _api_config_from_environment()

    lowest_effort = api.reasoning_supported_efforts[0]
    return api.model_copy(
        update={
            "reasoning_effort": lowest_effort,
            "max_retries": min(api.max_retries, 1),
            "timeout_seconds": min(api.timeout_seconds, 120),
        }
    )


def _load_api_config(path: Path) -> ApiConfig:
    try:
        return load_config(path).api
    except ConfigurationError as error:
        pytest.fail(f"live API configuration is invalid: {error}")


def _api_config_from_environment() -> ApiConfig:
    api_key = os.environ.get("REPORIVET_LIVE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("REPORIVET_LIVE_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
    model = os.environ.get("REPORIVET_LIVE_MODEL") or os.environ.get("OPENAI_MODEL")
    if not api_key or not model:
        pytest.fail(
            "live API tests require REPORIVET_LIVE_API_KEY and REPORIVET_LIVE_MODEL "
            "when no TOML configuration is available"
        )
    raw_context_window = os.environ.get("REPORIVET_LIVE_CONTEXT_WINDOW") or "128000"
    try:
        context_window = int(raw_context_window)
    except ValueError:
        pytest.fail("REPORIVET_LIVE_CONTEXT_WINDOW must be an integer")
    supported = tuple(
        value.strip()
        for value in os.environ.get("REPORIVET_LIVE_REASONING_EFFORTS", "low").split(",")
        if value.strip()
    )
    try:
        return ApiConfig(
            api_key=SecretStr(api_key),
            base_url=base_url,
            model=model,
            context_window_tokens=context_window,
            thinking_mode=os.environ.get("REPORIVET_LIVE_THINKING_MODE", "provider_default"),
            reasoning_effort=supported[-1] if supported else "low",
            reasoning_supported_efforts=supported or ("low",),
            timeout_seconds=float(os.environ.get("REPORIVET_LIVE_TIMEOUT_SECONDS", "90")),
            max_retries=1,
        )
    except (ValidationError, ValueError) as error:
        pytest.fail(f"live API environment configuration is invalid: {error}")
