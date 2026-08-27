"""Provider-aware and conservative provider-independent Token estimators."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import tiktoken
from tiktoken import Encoding

from repo_rivet.memory.token_calibrator import UsageCalibrator

CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
OPAQUE_PATTERN = re.compile(r"(?:[A-Za-z0-9+/=_-]{48,})")


class TokenEstimator(Protocol):
    """Estimate model request size without assuming one provider's message template."""

    @property
    def name(self) -> str:
        """Return a diagnostic estimator name."""
        ...

    def estimate_text(self, text: str, kind: str = "natural") -> int: ...

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int: ...


@dataclass
class ApproximateTokenEstimator:
    """Conservative, deterministic fallback for unknown provider tokenizers."""

    safety_factor: float = 1.20
    _tool_schema_cache: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return "approximate"

    def estimate_text(self, text: str, kind: str = "natural") -> int:
        if not text:
            return 0

        cjk_count = len(CJK_PATTERN.findall(text))
        non_cjk_count = len(text) - cjk_count
        chars_per_token = {
            "natural": 4.0,
            "code": 3.0,
            "json": 3.0,
            "log": 3.0,
            "opaque": 1.0,
        }.get(kind, 3.0)
        base_estimate = cjk_count + math.ceil(non_cjk_count / chars_per_token)

        opaque_chars = sum(len(match.group(0)) for match in OPAQUE_PATTERN.finditer(text))
        if opaque_chars:
            normal_chars = max(0, len(text) - opaque_chars)
            opaque_estimate = opaque_chars + math.ceil(normal_chars / chars_per_token)
            base_estimate = max(base_estimate, opaque_estimate)

        return max(1, math.ceil(base_estimate * self.safety_factor))

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        total = 0
        for message in messages:
            total += 8
            total += self.estimate_text(str(message.get("role", "")), kind="json")

            content = message.get("content")
            if isinstance(content, str):
                kind = "log" if message.get("role") == "tool" else "natural"
                total += self.estimate_text(content, kind=kind)

            remaining = {
                key: value
                for key, value in message.items()
                if key not in {"role", "content"} and value is not None
            }
            if remaining:
                total += self.estimate_text(_serialize(remaining), kind="json")

        total += self._estimate_tools(tools)
        return total

    def _estimate_tools(self, tools: list[dict[str, Any]]) -> int:
        if not tools:
            return 0
        serialized = _serialize(tools)
        cached = self._tool_schema_cache.get(serialized)
        if cached is None:
            cached = self.estimate_text(serialized, kind="json")
            self._tool_schema_cache = {serialized: cached}
        return cached


@dataclass
class ProviderTokenizerEstimator:
    """Use a known BPE tokenizer while leaving hidden message framing to calibration."""

    encoding: Encoding
    _tool_schema_cache: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"tokenizer:{self.encoding.name}"

    @classmethod
    def create(cls, model: str, encoding_name: str | None = None) -> ProviderTokenizerEstimator:
        encoding = (
            tiktoken.get_encoding(encoding_name)
            if encoding_name is not None
            else tiktoken.encoding_for_model(model)
        )
        return cls(encoding)

    def estimate_text(self, text: str, kind: str = "natural") -> int:
        del kind
        return len(self.encoding.encode(text, disallowed_special=()))

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        # Eight tokens per message is conservative framing headroom; actual provider
        # templates are learned by UsageCalibrator rather than assumed to be universal.
        message_tokens = sum(
            8 + self.estimate_text(_serialize(message), kind="json") for message in messages
        )
        return message_tokens + self._estimate_tools(tools)

    def _estimate_tools(self, tools: list[dict[str, Any]]) -> int:
        if not tools:
            return 0
        serialized = _serialize(tools)
        cached = self._tool_schema_cache.get(serialized)
        if cached is None:
            cached = self.estimate_text(serialized, kind="json")
            self._tool_schema_cache = {serialized: cached}
        return cached


@dataclass
class CalibratedTokenEstimator:
    """Apply provider-observed correction to any raw estimator."""

    base: TokenEstimator
    calibrator: UsageCalibrator

    @property
    def name(self) -> str:
        return f"calibrated:{self.base.name}"

    def estimate_text(self, text: str, kind: str = "natural") -> int:
        raw = self.base.estimate_text(text, kind=kind)
        return math.ceil(raw * self.calibrator.correction_factor())

    def raw_estimate_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        return self.base.estimate_request(messages, tools)

    def estimate_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        raw = self.raw_estimate_request(messages, tools)
        return math.ceil(raw * self.calibrator.correction_factor())


def create_token_estimator(
    *,
    model: str,
    tokenizer_encoding: str | None,
) -> TokenEstimator:
    """Prefer a provider tokenizer but always retain a no-network fallback."""
    try:
        return ProviderTokenizerEstimator.create(model, tokenizer_encoding)
    except Exception:
        return ApproximateTokenEstimator()


def _serialize(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
        sort_keys=True,
    )
