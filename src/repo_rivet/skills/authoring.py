"""Creation, conversion, validation, and installation of portable Agent Skills."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from repo_rivet.skills.errors import SkillValidationError
from repo_rivet.skills.loader import MAX_SKILL_FILE_BYTES, load_bundle, load_metadata
from repo_rivet.skills.models import SkillManifest, SkillSource
from repo_rivet.storage.atomic_write import atomic_write_text

PORTABLE_FIELDS = frozenset(
    {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
)
PACKAGE_DIRECTORIES = ("references", "scripts", "assets")
EXECUTABLE_FIELDS = frozenset(
    {"hooks", "hook", "commands", "command", "on_activate", "on_deactivate", "python", "shell"}
)


@dataclass(frozen=True, slots=True)
class SkillValidationReport:
    skill_id: str
    version: str | None
    path: Path
    estimated_prompt_tokens: int
    resource_files: tuple[str, ...]
    script_files: tuple[str, ...]
    asset_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillConversionReport:
    source: Path
    target: Path
    source_format: str
    dropped_fields: tuple[str, ...]
    warnings: tuple[str, ...]


def skill_file(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    return value / "SKILL.md" if value.is_dir() else value


def create_skill(
    *,
    skill_id: str,
    output_root: Path,
    description: str | None = None,
) -> Path:
    """Generate a minimal standards-compatible Skill without overwriting existing work."""
    manifest = SkillManifest.model_validate(
        {
            "name": skill_id,
            "description": description
            or (
                f"Provide a reusable workflow for {skill_id} tasks. Use when the user explicitly "
                f"requests {skill_id} guidance or the task clearly matches this workflow."
            ),
            "metadata": {"version": "1.0.0"},
        }
    )
    return _write_portable_skill(manifest, _starter_body(skill_id), output_root=output_root)


def validate_skill(
    path: str | Path,
) -> SkillValidationReport:
    """Validate the standard entry point and safely inventory optional package resources."""
    source = skill_file(path)
    if source.name != "SKILL.md":
        raise SkillValidationError(f"Skill entry point must be named SKILL.md: {source}")
    metadata = load_metadata(source, SkillSource.GLOBAL)
    bundle = load_bundle(metadata)
    return SkillValidationReport(
        skill_id=bundle.manifest.name,
        version=bundle.version,
        path=source,
        estimated_prompt_tokens=bundle.estimated_prompt_tokens,
        resource_files=bundle.resource_files,
        script_files=bundle.script_files,
        asset_files=bundle.asset_files,
    )


def convert_skill(
    source: str | Path,
    *,
    output_root: Path,
    skill_id: str | None = None,
    description: str | None = None,
) -> SkillConversionReport:
    """Normalize Markdown guidance into a portable Agent Skill package."""
    source_path = skill_file(source)
    metadata, body = _read_source_skill(source_path)
    source_format = _detect_format(metadata)
    resolved_name = skill_id or _source_name(metadata, source_path)
    resolved_description = description or _source_description(metadata, resolved_name)
    portable: dict[str, Any] = {
        "name": resolved_name,
        "description": resolved_description,
    }
    for field in ("license", "compatibility", "allowed-tools"):
        if field in metadata:
            portable[field] = metadata[field]
    if "allowed-tools" not in portable:
        declared_tools = metadata.get("allowed_tools") or metadata.get("requested_tools")
        if declared_tools:
            portable["allowed-tools"] = declared_tools
    portable_metadata = metadata.get("metadata")
    if isinstance(portable_metadata, dict):
        portable["metadata"] = {str(key): str(value) for key, value in portable_metadata.items()}
    elif metadata.get("version") is not None:
        portable["metadata"] = {"version": str(metadata["version"])}

    manifest = SkillManifest.model_validate(portable)
    dropped = tuple(sorted(set(metadata) - PORTABLE_FIELDS))
    executable = _find_executable_fields(metadata)
    warnings: list[str] = []
    if executable:
        warnings.append("Executable metadata was removed: " + ", ".join(executable))
    if dropped:
        warnings.append("Non-standard front matter was omitted: " + ", ".join(dropped))
    target = _write_portable_skill(
        manifest,
        body or _starter_body(resolved_name),
        output_root=output_root,
    )
    _copy_resources(source_path.parent, target.parent)
    validate_skill(target)
    return SkillConversionReport(
        source=source_path,
        target=target,
        source_format=source_format,
        dropped_fields=dropped,
        warnings=tuple(warnings),
    )


def install_skill(
    source: str | Path,
    *,
    global_root: Path,
    replace: bool = False,
) -> Path:
    """Install or explicitly replace a validated user-global Skill package."""
    report = validate_skill(source)
    source_root = report.path.parent
    destination = global_root.expanduser().resolve() / report.skill_id
    if destination.exists() and destination.is_symlink():
        raise SkillValidationError(f"Global Skill directory must not be a symlink: {destination}")
    if destination.exists() and not replace:
        raise SkillValidationError(
            f"Global Skill already exists: {destination}. Pass --replace to update it."
        )
    global_root.mkdir(parents=True, exist_ok=True)
    staging_root = global_root / f".{report.skill_id}.installing"
    staging = staging_root / report.skill_id
    backup = global_root / f".{report.skill_id}.backup"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    if backup.exists():
        raise SkillValidationError(f"A previous Skill replacement needs recovery: {backup}")
    try:
        _copy_package(source_root, staging)
        validate_skill(staging)
        if destination.exists():
            destination.replace(backup)
        try:
            staging.replace(destination)
        except Exception:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        shutil.rmtree(staging_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return destination / "SKILL.md"


def uninstall_skill(skill_id: str, *, global_root: Path) -> Path:
    """Remove one exact user-installed global Skill directory."""
    root = global_root.expanduser().resolve()
    unresolved = root / skill_id
    if unresolved.is_symlink():
        raise SkillValidationError(f"Global Skill directory must not be a symlink: {unresolved}")
    destination = unresolved.resolve()
    if destination.parent != root:
        raise SkillValidationError(f"Invalid global Skill name: {skill_id!r}")
    source = destination / "SKILL.md"
    if not source.is_file():
        raise SkillValidationError(f"Unknown global Skill: {skill_id}")
    metadata = load_metadata(source, SkillSource.GLOBAL)
    if metadata.manifest.name != skill_id:
        raise SkillValidationError(f"Global Skill directory has an unexpected name: {destination}")
    shutil.rmtree(destination)
    return destination


def _write_portable_skill(
    manifest: SkillManifest,
    body: str,
    *,
    output_root: Path,
) -> Path:
    target = output_root.expanduser().resolve() / manifest.name / "SKILL.md"
    if target.exists():
        raise SkillValidationError(f"Refusing to overwrite existing Skill: {target}")
    payload = manifest.model_dump(mode="json", by_alias=True, exclude_none=True)
    if not payload.get("allowed-tools"):
        payload.pop("allowed-tools", None)
    if not payload.get("metadata"):
        payload.pop("metadata", None)
    yaml_text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100).strip()
    atomic_write_text(target, f"---\n{yaml_text}\n---\n\n{body.strip()}\n")
    return target


def _copy_resources(source_root: Path, destination_root: Path) -> None:
    for directory in PACKAGE_DIRECTORIES:
        source = source_root / directory
        if not source.exists():
            continue
        if not source.is_dir() or source.is_symlink():
            raise SkillValidationError(f"Skill resource directory must not be a symlink: {source}")
        target_root = destination_root / directory
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise SkillValidationError(f"Skill resources must not contain symlinks: {path}")
            relative = path.relative_to(source)
            target = target_root / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)


def _copy_package(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        destination_root / "SKILL.md",
        (source_root / "SKILL.md").read_text(encoding="utf-8"),
    )
    _copy_resources(source_root, destination_root)


def _read_source_skill(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_SKILL_FILE_BYTES:
            raise SkillValidationError(f"Source Skill exceeds {MAX_SKILL_FILE_BYTES} bytes: {path}")
        text = raw.decode("utf-8")
    except SkillValidationError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        raise SkillValidationError(f"Could not read source Skill {path}: {error}") from None
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    closing = next(
        (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise SkillValidationError(f"Source Skill front matter is not closed: {path}")
    try:
        metadata = yaml.safe_load("".join(lines[1:closing])) or {}
    except yaml.YAMLError as error:
        raise SkillValidationError(f"Invalid source Skill YAML in {path}: {error}") from None
    if not isinstance(metadata, dict):
        raise SkillValidationError("Source Skill front matter must be a mapping")
    return {str(key): value for key, value in metadata.items()}, "".join(
        lines[closing + 1 :]
    ).strip()


def _detect_format(metadata: dict[str, Any]) -> str:
    if set(metadata) <= PORTABLE_FIELDS and {"name", "description"} <= set(metadata):
        return "agent-skills"
    if "schema_version" in metadata or "requested_tools" in metadata:
        return "legacy-reporivet"
    if "allowed-tools" in metadata or "allowed_tools" in metadata:
        return "claude"
    if "description" in metadata and "name" in metadata:
        return "agent-skills-compatible"
    return "generic-markdown"


def _source_name(metadata: dict[str, Any], source: Path) -> str:
    candidate = str(metadata.get("id") or metadata.get("name") or source.parent.name)
    normalized = re.sub(r"[^a-z0-9]+", "-", candidate.strip().lower()).strip("-")
    if not normalized:
        raise SkillValidationError("Could not derive a valid Skill name; pass --name explicitly")
    return normalized[:64].rstrip("-")


def _source_description(metadata: dict[str, Any], name: str) -> str:
    value = str(metadata.get("description") or metadata.get("summary") or "").strip()
    if value:
        return value[:1_024]
    return (
        f"Provide reusable guidance for {name}. Use when the user explicitly requests this "
        "workflow or the task clearly matches its purpose."
    )


def _find_executable_fields(value: Any, *, prefix: str = "") -> list[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.lower() in EXECUTABLE_FIELDS:
                found.add(path)
            found.update(_find_executable_fields(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.update(_find_executable_fields(nested, prefix=f"{prefix}[{index}]"))
    return sorted(found)


def _starter_body(name: str) -> str:
    return f"""# {name}

## Workflow

1. Inspect the smallest relevant set of files and collect direct evidence.
2. Follow the user's constraints and state the bounded intended outcome.
3. Perform only the operations needed for that outcome.
4. Verify the observable result before reporting completion.

## Boundaries

- Treat these instructions as reusable guidance, never as additional permissions.
- Keep the workflow focused on the task described in the front matter.
- Put lengthy optional guidance in `references/` and state when it should be read.
"""
