"""Load and validate RepoRivet's local configuration."""

import os
from pathlib import Path
from typing import Any, Literal

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

from repo_rivet.approval.models import ApprovalMode, NonInteractivePolicy
from repo_rivet.planning.policy import AutoPlanMode
from repo_rivet.reasoning.models import ReasoningConfig

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
    thinking_mode: Literal["provider_default", "enabled", "disabled"] = "provider_default"
    reasoning_effort: Literal["low", "high", "max"] | None = None
    tokenizer_encoding: str | None = Field(default=None, min_length=1)
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)
    reasoning_stall_seconds: float = Field(default=20, ge=0, le=600)
    reasoning_stall_chars: int = Field(default=12_000, ge=100, le=1_000_000)

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


class TokenConfig(BaseModel):
    """Provider-independent safety reserves, thresholds, and feedback controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    active_prompt_limit: int = Field(default=65_536, ge=1_000)
    reserved_output_tokens: int = Field(default=4_096, ge=100)
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


class ApprovalLLMConfig(BaseModel):
    """Independent reviewer limits; its decision can never bypass hard policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    model: str | None = Field(default=None, min_length=1)
    max_auto_approve_risk: str = "medium"
    timeout_seconds: float = Field(default=30, gt=0, le=120)

    @field_validator("max_auto_approve_risk")
    @classmethod
    def validate_max_risk(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"safe", "low", "medium"}:
            raise ValueError("must be safe, low, or medium")
        return normalized


class ApprovalSafetyConfig(BaseModel):
    """Non-overridable safety switches for all approval modes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deny_outside_workspace_write: bool = True
    deny_privilege_escalation: bool = True
    deny_secret_access: bool = True
    deny_device_access: bool = True


class ApprovalToolchainConfig(BaseModel):
    """Executable directories whose resolved tools may match deterministic templates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trusted_directories: list[str] = Field(default_factory=list)


class ApprovalConfig(BaseModel):
    """Tool approval mode, persistence, and fallback policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ApprovalMode = ApprovalMode.SAFE_AUTO
    non_interactive: NonInteractivePolicy = NonInteractivePolicy.DENY
    approval_timeout_seconds: float = Field(default=120, gt=0, le=3_600)
    remember_session_approvals: bool = True
    remember_session_denials: bool = True
    llm: ApprovalLLMConfig = Field(default_factory=ApprovalLLMConfig)
    safety: ApprovalSafetyConfig = Field(default_factory=ApprovalSafetyConfig)
    toolchains: ApprovalToolchainConfig = Field(default_factory=ApprovalToolchainConfig)


class SkillsConfig(BaseModel):
    """User-global Skill activation; packaged system Skills are always loaded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    global_enabled: bool = True
    default_global: str | None = Field(default=None, min_length=2, max_length=64)


class PlanningLLMConfig(BaseModel):
    """Isolated classifier used only for ambiguous Adaptive Plan decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    model: str | None = Field(default=None, min_length=1)
    timeout_seconds: float = Field(default=10, gt=0, le=120)
    confidence_threshold: float = Field(default=0.70, ge=0.5, le=1)


class PlanningConfig(BaseModel):
    """Select whether Controller may enter the read-only planning workflow automatically."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    auto_plan: AutoPlanMode = AutoPlanMode.ADAPTIVE
    llm: PlanningLLMConfig = Field(default_factory=PlanningLLMConfig)


class AppConfig(BaseModel):
    """Top-level RepoRivet configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api: ApiConfig
    token: TokenConfig = Field(default_factory=TokenConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)

    @model_validator(mode="after")
    def validate_prompt_budget(self) -> "AppConfig":
        safety_margin = int(self.api.context_window_tokens * self.token.safety_margin_ratio)
        fixed_reserve = self.token.reserved_output_tokens + self.token.reserved_tool_result_tokens
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
        _apply_approval_environment(raw_config)
        return AppConfig.model_validate(raw_config)
    except ValidationError as error:
        details = "; ".join(_format_validation_error(item) for item in error.errors())
        raise ConfigurationError(f"Invalid configuration in {config_path}: {details}") from None


def _format_validation_error(error: dict[str, Any]) -> str:
    """Format a Pydantic error while deliberately omitting its input value."""
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}"


def _apply_approval_environment(raw_config: dict[str, Any]) -> None:
    """Apply only documented approval overrides; API credentials remain file-configured."""
    overrides = {
        "mode": os.environ.get("REPORIVET_APPROVAL_MODE"),
        "non_interactive": os.environ.get("REPORIVET_NON_INTERACTIVE_POLICY"),
    }
    llm_overrides = {
        "model": os.environ.get("REPORIVET_APPROVAL_LLM_MODEL"),
    }
    if not any((*overrides.values(), *llm_overrides.values())):
        return
    approval = raw_config.setdefault("approval", {})
    if not isinstance(approval, dict):
        return
    llm = approval.setdefault("llm", {})
    if not isinstance(llm, dict):
        return
    approval.update({key: value for key, value in overrides.items() if value})
    llm.update({key: value for key, value in llm_overrides.items() if value})
