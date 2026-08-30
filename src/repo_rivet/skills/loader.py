"""Safe Agent Skill discovery with lazy instruction loading."""

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
MAX_FRONT_MATTER_BYTES = 64_000
MAX_INSTRUCTION_TOKENS = 5_000


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(65_536):
                digest.update(chunk)
    except OSError as error:
        raise SkillValidationError(f"Could not read skill file {path}: {error}") from None
    return digest.hexdigest()


def _read_front_matter(path: Path) -> tuple[dict[str, Any], int]:
    """Read only the bounded YAML header and return the body byte offset."""
    try:
        size = path.stat().st_size
        if size > MAX_SKILL_FILE_BYTES:
            raise SkillValidationError(f"Skill file exceeds {MAX_SKILL_FILE_BYTES} bytes: {path}")
        with path.open("rb") as stream:
            first = stream.readline()
            if first.strip() != b"---":
                raise SkillValidationError(f"Skill file must begin with YAML front matter: {path}")
            header: list[bytes] = []
            consumed = len(first)
            while True:
                line = stream.readline()
                if not line:
                    raise SkillValidationError(f"Skill front matter is not closed: {path}")
                consumed += len(line)
                if consumed > MAX_FRONT_MATTER_BYTES:
                    raise SkillValidationError(
                        f"Skill front matter exceeds {MAX_FRONT_MATTER_BYTES} bytes: {path}"
                    )
                if line.strip() == b"---":
                    break
                header.append(line)
    except SkillValidationError:
        raise
    except OSError as error:
        raise SkillValidationError(f"Could not read skill file {path}: {error}") from None
    try:
        text = b"".join(header).decode("utf-8")
        front_matter = yaml.safe_load(text)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise SkillValidationError(f"Invalid YAML front matter in {path}: {error}") from None
    if not isinstance(front_matter, dict):
        raise SkillValidationError(f"Skill front matter must be a mapping: {path}")
    return {str(key): value for key, value in front_matter.items()}, consumed


def load_metadata(path: Path, source: SkillSource) -> SkillMetadata:
    """Index portable metadata without loading the Markdown instruction body."""
    front_matter, body_offset = _read_front_matter(path)
    try:
        manifest = SkillManifest.model_validate(front_matter)
    except ValidationError as error:
        raise SkillValidationError(f"Invalid skill manifest in {path}: {error}") from None
    if path.parent.name != manifest.name:
        raise SkillValidationError(
            f"Skill directory name must match name {manifest.name!r}: {path.parent.name!r}"
        )
    manifest_hash = _hash_file(path)
    body_size = max(0, path.stat().st_size - body_offset)
    if body_size == 0:
        raise SkillValidationError(f"Skill body must not be empty: {path}")
    return SkillMetadata(
        qualified_id=f"{source.value}:{manifest.name}",
        manifest=manifest,
        source=source,
        path=path,
        manifest_hash=manifest_hash,
        body_size_bytes=body_size,
    )


def _inventory(root: Path, directory: str) -> tuple[str, ...]:
    base = root / directory
    if not base.is_dir() or base.is_symlink():
        return ()
    files: list[str] = []
    resolved_root = root.resolve()
    for path in sorted(base.rglob("*")):
        if path.is_file() and not path.is_symlink():
            resolved = path.resolve()
            if resolved.is_relative_to(resolved_root):
                files.append(path.relative_to(root).as_posix())
    return tuple(files)


def load_bundle(metadata: SkillMetadata) -> SkillBundle:
    """Load instructions only after a descriptor has been selected."""
    current = load_metadata(metadata.path, metadata.source)
    if current != metadata:
        raise SkillValidationError(
            f"Skill metadata changed during activation: {metadata.qualified_id}"
        )
    try:
        text = metadata.path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SkillValidationError(
            f"Could not read UTF-8 skill file {metadata.path}: {error}"
        ) from None
    lines = text.splitlines(keepends=True)
    closing = next(
        (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"),
        None,
    )
    assert closing is not None
    body = "".join(lines[closing + 1 :]).strip()
    if not body:
        raise SkillValidationError(f"Skill body must not be empty: {metadata.path}")
    estimated = ApproximateTokenEstimator(safety_factor=1.0).estimate_text(body)
    if estimated > MAX_INSTRUCTION_TOKENS:
        raise SkillValidationError(
            f"Skill {metadata.qualified_id} body exceeds "
            f"{MAX_INSTRUCTION_TOKENS} instruction tokens"
        )
    root = metadata.path.parent
    return SkillBundle(
        **metadata.model_dump(),
        body=body,
        content_hash=metadata.manifest_hash,
        estimated_prompt_tokens=estimated,
        resource_files=_inventory(root, "references"),
        script_files=_inventory(root, "scripts"),
        asset_files=_inventory(root, "assets"),
    )
