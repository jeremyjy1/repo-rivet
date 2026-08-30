"""Classify the small set of failures safe for Controller-owned retries."""

from repo_rivet.actions.models import RetryClass
from repo_rivet.tools.base import ToolResult

_TRANSIENT_ERROR_CODES = frozenset(
    {
        "io_temporarily_unavailable",
        "process_spawn_transient",
        "rate_limited",
        "temporarily_unavailable",
        "tool_busy",
    }
)


def retry_class_for(result: ToolResult) -> RetryClass:
    if result.retryable and result.error_code in _TRANSIENT_ERROR_CODES:
        return RetryClass.TRANSIENT_INFRASTRUCTURE
    if result.ok:
        return RetryClass.NONE
    return RetryClass.BUSINESS_FAILURE


def may_retry_internally(result: ToolResult, *, attempt: int, max_attempts: int = 2) -> bool:
    return retry_class_for(result) == RetryClass.TRANSIENT_INFRASTRUCTURE and attempt < max_attempts
