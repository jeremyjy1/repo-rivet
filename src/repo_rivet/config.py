"""Load and validate RepoRivet's local configuration."""

from pathlib import Path
from typing import Any

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic.functional_validators import field_validator

DEFAULT_CONFIG_PATH = Path("reporivet.toml")
_API_KEY_PLACEHOLDER = "replace-with-your-api-key"
_MODEL_PLACEHOLDER = "replace-with-model-name"


class ConfigurationError(ValueError):
    """Raised when the local configuration cannot be loaded safely."""


class ApiConfig(BaseModel):
    """Settings for an OpenAI-compatible API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr
    base_url: AnyHttpUrl
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, value: Any) -> str:
        """Reject missing, blank, or unchanged example credentials."""
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")

        value = value.strip()
        if value == _API_KEY_PLACEHOLDER:
            raise ValueError("must be replaced with a real API key")
        return value

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, value: Any) -> str:
        """Normalize and validate the model name."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        value = value.strip()
        if value == _MODEL_PLACEHOLDER:
            raise ValueError("must be replaced with a real model name")
        return value


class AppConfig(BaseModel):
    """Top-level RepoRivet configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api: ApiConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load a TOML configuration without exposing its values in errors."""
    import tomllib

    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
    except FileNotFoundError:
        raise ConfigurationError(
            f"Configuration file not found: {config_path}. "
            "Copy reporivet.example.toml to reporivet.toml and fill in the API settings."
        ) from None
    except OSError as error:
        raise ConfigurationError(f"Could not read configuration file: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(
            f"Invalid TOML in configuration file: {config_path}: {error}"
        ) from None

    try:
        return AppConfig.model_validate(raw_config)
    except ValidationError as error:
        details = "; ".join(_format_validation_error(item) for item in error.errors())
        raise ConfigurationError(f"Invalid configuration in {config_path}: {details}") from None


def _format_validation_error(error: dict[str, Any]) -> str:
    """Format a Pydantic error while deliberately omitting its input value."""
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}"
