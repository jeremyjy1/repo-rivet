"""Load and validate RepoRivet's local configuration."""

from pathlib import Path
from typing import Any

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)
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
    context_window_tokens: int = Field(ge=1_000)
    max_output_tokens: int = Field(default=4_096, ge=1)
    tokenizer_encoding: str | None = Field(default=None, min_length=1)
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

    @model_validator(mode="after")
    def validate_token_limits(self) -> "ApiConfig":
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("max_output_tokens must be smaller than context_window_tokens")
        return self


class TokenConfig(BaseModel):
    """Provider-independent safety reserves, thresholds, and feedback controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reserved_tool_result_tokens: int = Field(default=2_048, ge=0)
    safety_margin_ratio: float = Field(default=0.15, ge=0, lt=0.5)
    soft_limit_ratio: float = Field(default=0.70, gt=0, lt=1)
    hard_limit_ratio: float = Field(default=0.85, gt=0, le=1)
    default_correction_factor: float = Field(default=1.25, ge=1.0, le=3.0)
    calibration_window: int = Field(default=20, ge=1, le=100)
    max_context_overflow_retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "TokenConfig":
        if self.hard_limit_ratio <= self.soft_limit_ratio:
            raise ValueError("hard_limit_ratio must exceed soft_limit_ratio")
        return self


class AppConfig(BaseModel):
    """Top-level RepoRivet configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api: ApiConfig
    token: TokenConfig = Field(default_factory=TokenConfig)

    @model_validator(mode="after")
    def validate_prompt_budget(self) -> "AppConfig":
        safety_margin = int(self.api.context_window_tokens * self.token.safety_margin_ratio)
        fixed_reserve = self.api.max_output_tokens + self.token.reserved_tool_result_tokens
        total_reserve = fixed_reserve + safety_margin
        if total_reserve >= self.api.context_window_tokens:
            raise ValueError(
                "output, tool-result, and safety reserves must leave a positive prompt budget"
            )
        return self


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
