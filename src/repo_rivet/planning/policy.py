"""Controller-owned policy for entering the read-only planning workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class AutoPlanMode(StrEnum):
    OFF = "off"
    ADAPTIVE = "adaptive"
    ALWAYS = "always"


_HIGH_SCOPE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:across|touch(?:ing)?)\s+(?:multiple|several)\s+files\b",
        r"\b(?:large[- ]scale|project[- ]wide|repo(?:sitory)?[- ]wide)\s+refactor\b",
        r"\b(?:migrate|migration)\s+(?:the\s+)?(?:entire|whole|project|repository)\b",
        r"\b(?:new|complete)\s+(?:application|project|service)\s+from\s+scratch\b",
        r"(?:多文件|跨文件|整个项目|全项目|项目级|整体重构|架构迁移|从零(?:实现|创建|搭建))",
    )
)
_LIST_ITEM = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S")


@dataclass(frozen=True, slots=True)
class AutoPlanPolicy:
    """Choose planning conservatively; ambiguous cases stay model-requested."""

    mode: AutoPlanMode = AutoPlanMode.OFF
    classifier_confidence_threshold: float = 0.70

    def preflight_reason(self, task: str) -> str | None:
        normalized = task.strip()
        if self.mode == AutoPlanMode.OFF:
            return None
        if self.mode == AutoPlanMode.ALWAYS:
            return "auto-plan mode is always"
        if any(pattern.search(normalized) for pattern in _HIGH_SCOPE_PATTERNS):
            return "task description declares project-wide or multi-file scope"
        if len(_LIST_ITEM.findall(normalized)) >= 4:
            return "task contains at least four explicit work items"
        if len(normalized) >= 1_200:
            return "task specification is large enough to require scoped planning"
        return None

    @property
    def model_may_request(self) -> bool:
        return self.mode == AutoPlanMode.ADAPTIVE
