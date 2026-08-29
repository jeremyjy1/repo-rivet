"""Strict SKILL.md parser with bounded, lazy body loading."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from repo_rivet.memory.token_estimator import ApproximateTokenEstimator
from repo_rivet.skills.errors import SkillValidationError
from repo_rivet.skills.models import SkillBundle, SkillManifest, SkillMetadata, SkillSource

MAX_SKILL_FILE_BYTES = 256_000


def _read_parts(path: Path) -> tuple[dict[str, Any], str, bytes]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_SKILL_FILE_BYTES:
            raise SkillValidationError(f"Skill file exceeds {MAX_SKILL_FILE_BYTES} bytes: {path}")
        text = raw.decode("utf-8")
    except SkillValidationError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        raise SkillValidationError(f"Could not read UTF-8 skill file {path}: {error}") from None
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise SkillValidationError(f"Skill file must begin with YAML front matter: {path}")
    closing = next(
        (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise SkillValidationError(f"Skill front matter is not closed: {path}")
    try:
        front_matter = yaml.safe_load("".join(lines[1:closing]))
    except yaml.YAMLError as error:
        raise SkillValidationError(f"Invalid YAML front matter in {path}: {error}") from None
    if not isinstance(front_matter, dict):
        raise SkillValidationError(f"Skill front matter must be a mapping: {path}")
    body = "".join(lines[closing + 1 :]).strip()
    if not body:
        raise SkillValidationError(f"Skill body must not be empty: {path}")
    return front_matter, body, raw


def load_metadata(path: Path, source: SkillSource) -> SkillMetadata:
    """Load only validated front matter into the in-memory discovery index."""
    front_matter, _body, _raw = _read_parts(path)
    try:
        manifest = SkillManifest.model_validate(front_matter)
    except ValidationError as error:
        raise SkillValidationError(f"Invalid skill manifest in {path}: {error}") from None
    if path.parent.name != manifest.id:
        raise SkillValidationError(
            f"Skill directory name must match id {manifest.id!r}: {path.parent.name!r}"
        )
    return SkillMetadata(manifest=manifest, source=source, path=path)


def load_bundle(metadata: SkillMetadata) -> SkillBundle:
    """Load the Markdown body for runtime system loading or global selection."""
    front_matter, body, raw = _read_parts(metadata.path)
    try:
        manifest = SkillManifest.model_validate(front_matter)
    except ValidationError as error:
        raise SkillValidationError(f"Invalid skill manifest in {metadata.path}: {error}") from None
    if manifest != metadata.manifest:
        raise SkillValidationError(f"Skill metadata changed during activation: {manifest.id}")
    estimated = ApproximateTokenEstimator(safety_factor=1.0).estimate_text(body)
    if estimated > manifest.limits.max_prompt_tokens:
        raise SkillValidationError(
            f"Skill {manifest.id} body exceeds max_prompt_tokens "
            f"({estimated} > {manifest.limits.max_prompt_tokens})"
        )
    return SkillBundle(
        manifest=manifest,
        source=metadata.source,
        path=metadata.path,
        body=body,
        content_hash=hashlib.sha256(raw).hexdigest(),
        estimated_prompt_tokens=estimated,
    )
