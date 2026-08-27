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
timeout_seconds = 30
max_retries = 2
""",
    )

    config = load_config(config_path)

    assert isinstance(config.api.api_key, SecretStr)
    assert config.api.api_key.get_secret_value() == "test-secret"
    assert str(config.api.base_url) == "https://example.com/v1"
    assert config.api.model == "test-model"
    assert config.api.timeout_seconds == 30
    assert config.api.max_retries == 2


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
unexpected = true
""",
    )

    with pytest.raises(ConfigurationError, match="api.unexpected"):
        load_config(config_path)


def test_load_config_rejects_invalid_toml(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "reporivet.toml", "[api\n")

    with pytest.raises(ConfigurationError, match="Invalid TOML"):
        load_config(config_path)
