"""Canonicalize tool requests for analysis and exact-session grants."""

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

from repo_rivet.approval.models import (
    ApprovalRequest,
    Capability,
    RiskAssessment,
    RiskLevel,
)

POLICY_VERSION = "approval-v1"
_PATH_KEYS: dict[str, tuple[str, ...]] = {
    "list_files": ("path",),
    "search_text": ("path",),
    "read_file": ("path",),
    "write_file": ("path",),
    "edit_file": ("path",),
    "run_command": ("cwd",),
    "run_verification": ("cwd",),
    "git_diff": ("path",),
    "git_status": ("path",),
}
_CONTENT_KEYS = frozenset({"content", "stdin"})
_SECRET_ARGUMENT_PATTERN = re.compile(
    r"(?i)(authorization|api[_-]?key|password|secret|token)(?:\s*[:=]|$)"
)
_SECRET_VALUE_FLAGS = frozenset(
    {"--api-key", "--authorization", "--password", "--secret", "--token"}
)


def request_fingerprint(
    tool_name: str,
    normalized_arguments: dict[str, Any],
    workspace: str,
    policy_version: str = POLICY_VERSION,
) -> str:
    payload = {
        "tool": tool_name,
        "arguments": normalized_arguments,
        "workspace": workspace,
        "policy_version": policy_version,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class RequestNormalizer:
    """Resolve paths and canonicalize opaque content without logging its value."""

    def __init__(self, workspace: str | Path, *, policy_version: str = POLICY_VERSION) -> None:
        self.workspace = Path(workspace).resolve(strict=True)
        self.policy_version = policy_version

    def normalize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str,
        declared_capabilities: set[Capability] | frozenset[Capability],
    ) -> ApprovalRequest:
        normalized = self._normalize_arguments(tool_name, arguments)
        fingerprint = request_fingerprint(
            tool_name,
            normalized,
            str(self.workspace),
            self.policy_version,
        )
        return ApprovalRequest(
            request_id=f"approval-{uuid4().hex[:12]}",
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            normalized_arguments=normalized,
            declared_capabilities=set(declared_capabilities),
            workspace=str(self.workspace),
            fingerprint=fingerprint,
            assessment=RiskAssessment(level=RiskLevel.MEDIUM),
        )

    def refresh(self, request: ApprovalRequest) -> ApprovalRequest:
        return self.normalize(
            tool_name=request.tool_name,
            arguments=request.arguments,
            session_id=request.session_id,
            declared_capabilities=request.declared_capabilities,
        )

    def _normalize_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in arguments.items():
            if key in _CONTENT_KEYS and isinstance(value, str):
                normalized[key] = {
                    "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                    "characters": len(value),
                }
            else:
                normalized[key] = value

        resolved_paths: dict[str, str] = {}
        outside_paths: list[str] = []
        for key in _PATH_KEYS.get(tool_name, ()):
            raw_path = arguments.get(key, ".")
            if not isinstance(raw_path, str):
                continue
            requested = Path(raw_path).expanduser()
            target = (
                requested.resolve(strict=False)
                if requested.is_absolute()
                else (self.workspace / requested).resolve(strict=False)
            )
            resolved_paths[key] = str(target)
            if not target.is_relative_to(self.workspace):
                outside_paths.append(str(target))
        if resolved_paths:
            normalized["_resolved_paths"] = resolved_paths
        if outside_paths:
            normalized["_outside_workspace_paths"] = sorted(outside_paths)

        if tool_name in {"run_command", "run_verification"}:
            command = arguments.get("command")
            if isinstance(command, str):
                try:
                    argv = shlex.split(command, posix=True)
                except ValueError:
                    argv = [command]
                normalized["command"] = _normalize_command(argv)
        return normalized


def _normalize_command(argv: list[str]) -> dict[str, Any]:
    if not argv:
        return {"program": "", "args": []}
    safe_args: list[Any] = []
    redact_next = False
    for argument in argv[1:]:
        lower = argument.lower()
        should_redact = redact_next or bool(_SECRET_ARGUMENT_PATTERN.search(argument))
        if should_redact:
            safe_args.append(
                {
                    "redacted": True,
                    "sha256": hashlib.sha256(argument.encode("utf-8")).hexdigest(),
                    "characters": len(argument),
                }
            )
        else:
            safe_args.append(argument)
        redact_next = lower in _SECRET_VALUE_FLAGS or lower in {"-h", "--header"}
    return {"program": argv[0], "args": safe_args}
