from pathlib import Path

import pytest
from pydantic import SecretStr

from repo_rivet.config import ConfigurationError, load_config


def write_config(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "reporivet.toml",
        """
[api]
api_key = "test-secret"
base_url = "https://example.com/v1"
model = "test-model"
context_window_tokens = 32768
tokenizer_encoding = "cl100k_base"
timeout_seconds = 30
max_retries = 2

[token]
active_prompt_limit = 16384
reserved_output_tokens = 2048
reserved_tool_result_tokens = 1024
safety_margin_ratio = 0.10
soft_limit_ratio = 0.65
hard_limit_ratio = 0.80
default_correction_factor = 1.30
calibration_window = 10
max_context_overflow_retries = 1
""",
    )

    config = load_config(config_path)

    assert isinstance(config.api.api_key, SecretStr)
    assert config.api.api_key.get_secret_value() == "test-secret"
    assert str(config.api.base_url) == "https://example.com/v1"
    assert config.api.model == "test-model"
    assert config.api.context_window_tokens == 32768
    assert config.api.tokenizer_encoding == "cl100k_base"
    assert config.api.timeout_seconds == 30
    assert config.api.max_retries == 2
    assert config.token.active_prompt_limit == 16384
    assert config.token.reserved_output_tokens == 2048
    assert config.token.reserved_tool_result_tokens == 1024
    assert config.token.safety_margin_ratio == 0.10
    assert config.token.soft_limit_ratio == 0.65
    assert config.token.hard_limit_ratio == 0.80
    assert config.token.default_correction_factor == 1.30
    assert config.token.calibration_window == 10
    assert config.token.max_context_overflow_retries == 1
    assert config.approval.mode.value == "safe-auto"
    assert config.approval.non_interactive.value == "deny"
    assert config.skills.global_enabled
    assert config.skills.default_global is None
    assert config.planning.auto_plan.value == "adaptive"


def test_load_config_accepts_auto_plan_mode(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "reporivet.toml",
        """
[api]
api_key = "test-secret"
base_url = "https://example.com/v1"
model = "test-model"
context_window_tokens = 32768

[planning]
auto_plan = "always"
""",
    )

    assert load_config(config_path).planning.auto_plan.value == "always"


def test_load_config_accepts_approval_settings(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "reporivet.toml",
        """
[api]
api_key = "test-secret"
base_url = "https://example.com/v1"
model = "test-model"
context_window_tokens = 32768

[approval]
mode = "llm-auto"
non_interactive = "fail"
remember_session_approvals = false

[approval.llm]
enabled = true
model = "review-model"
max_auto_approve_risk = "low"

[approval.safety]
deny_secret_access = false

[reasoning]
display = "trace"
recent_event_limit = 12
max_summary_chars = 600
""",
    )

    config = load_config(config_path)

    assert config.approval.mode.value == "llm-auto"
    assert config.approval.non_interactive.value == "fail"
    assert not config.approval.remember_session_approvals
    assert config.approval.llm.model == "review-model"
    assert config.approval.llm.max_auto_approve_risk == "low"
    assert not config.approval.safety.deny_secret_access
    assert config.reasoning.display.value == "trace"
    assert config.reasoning.recent_event_limit == 12
    assert config.reasoning.max_summary_chars == 600


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "reporivet.toml"

    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        load_config(config_path)


def test_load_config_rejects_example_api_key_without_leaking_it(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "reporivet.toml",
        """
[api]
api_key = "replace-with-your-api-key"
base_url = "https://example.com/v1"
model = "test-model"
context_window_tokens = 32768
""",
    )

    with pytest.raises(ConfigurationError) as captured:
        load_config(config_path)

    message = str(captured.value)
    assert "api.api_key" in message
    assert "replace-with-your-api-key" not in message


def test_load_config_rejects_example_model_name(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "reporivet.toml",
        """
[api]
api_key = "test-secret"
base_url = "https://example.com/v1"
model = "replace-with-model-name"
context_window_tokens = 32768
""",
    )

    with pytest.raises(ConfigurationError, match="api.model"):
        load_config(config_path)


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "reporivet.toml",
        """
[api]
api_key = "test-secret"
base_url = "https://example.com/v1"
model = "test-model"
context_window_tokens = 32768
unexpected = true
""",
    )

    with pytest.raises(ConfigurationError, match="api.unexpected"):
        load_config(config_path)


def test_load_config_rejects_invalid_toml(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "reporivet.toml", "[api\n")

    with pytest.raises(ConfigurationError, match="Invalid TOML"):
        load_config(config_path)


def test_load_config_requires_context_window_size(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "reporivet.toml",
        """
[api]
api_key = "test-secret"
base_url = "https://example.com/v1"
model = "test-model"
""",
    )

    with pytest.raises(ConfigurationError, match="api.context_window_tokens"):
        load_config(config_path)


def test_load_config_rejects_reserves_that_consume_context(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "reporivet.toml",
        """
[api]
api_key = "test-secret"
base_url = "https://example.com/v1"
model = "test-model"
context_window_tokens = 4096

[token]
reserved_output_tokens = 4096
""",
    )

    with pytest.raises(ConfigurationError, match="must leave a positive prompt budget"):
        load_config(config_path)
