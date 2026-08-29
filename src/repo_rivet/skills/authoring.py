"""Deterministic generation, conversion, validation, and installation of Skills."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from repo_rivet.memory.token_estimator import ApproximateTokenEstimator
from repo_rivet.skills.errors import SkillValidationError
from repo_rivet.skills.loader import MAX_SKILL_FILE_BYTES, load_bundle, load_metadata
from repo_rivet.skills.models import SkillManifest, SkillSource
from repo_rivet.storage.atomic_write import atomic_write_text

FIXED_REQUIREMENTS = frozenset(
    {
        "target_snapshot_current",
        "target_range_seen",
        "plan_approved",
        "no_active_processes",
        "required_build_passed",
        "required_tests_passed",
        "required_behavior_checks_passed",
        "no_stale_verification",
        "git_diff_reviewed",
    }
)

FOREIGN_TOOL_ALIASES = {
    "read": "read_file",
    "readfile": "read_file",
    "grep": "search_text",
    "search": "search_text",
    "glob": "list_files",
    "ls": "list_files",
    "edit": "edit_file",
    "multiedit": "edit_file",
    "write": "write_file",
    "bash": "run_command",
    "shell": "run_command",
}

EXECUTABLE_FIELDS = frozenset(
    {
        "hooks",
        "hook",
        "scripts",
        "script",
        "commands",
        "command",
        "on_activate",
        "on_deactivate",
        "python",
        "shell",
    }
)

MAPPED_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "name",
        "version",
        "summary",
        "description",
        "category",
        "activation",
        "triggers",
        "compatible_modes",
        "requested_tools",
        "allowed-tools",
        "allowed_tools",
        "tools",
        "requirements",
        "verification_profiles",
        "limits",
    }
)


@dataclass(frozen=True, slots=True)
class SkillValidationReport:
    skill_id: str
    version: str
    path: Path
    estimated_prompt_tokens: int
    requested_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillConversionReport:
    source: Path
    target: Path
    source_format: str
    mapped_tools: tuple[str, ...]
    dropped_fields: tuple[str, ...]
    warnings: tuple[str, ...]


def skill_file(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    return value / "SKILL.md" if value.is_dir() else value


def create_skill(
    *,
    skill_id: str,
    output_root: Path,
    name: str | None = None,
    summary: str | None = None,
    requested_tools: list[str] | None = None,
    compatible_modes: list[str] | None = None,
    before_edit: list[str] | None = None,
    before_finish: list[str] | None = None,
    known_tools: set[str] | None = None,
) -> Path:
    """Generate one valid native Skill draft without overwriting existing work."""
    display_name = name.strip() if name else _title_from_id(skill_id)
    manifest = SkillManifest.model_validate(
        {
            "schema_version": 1,
            "id": skill_id,
            "name": display_name,
            "version": "1.0.0",
            "summary": (
                summary.strip() if summary else f"Reusable workflow for {display_name} tasks."
            ),
            "category": "workflow",
            "activation": {"explicit": True, "automatic": False},
            "triggers": {},
            "compatible_modes": compatible_modes or ["plan", "execute"],
            "requested_tools": requested_tools or ["list_files", "read_file", "search_text"],
            "requirements": {
                "before_edit": before_edit or [],
                "before_finish": before_finish or [],
            },
            "verification_profiles": [],
            "limits": {"max_prompt_tokens": 1400, "max_active_support_skills": 0},
        }
    )
    available_tools = manifest.requested_tools if known_tools is None else known_tools
    unknown_tools = manifest.requested_tools - available_tools
    if unknown_tools:
        raise SkillValidationError(
            "Cannot generate a Skill with unknown tools: " + ", ".join(sorted(unknown_tools))
        )
    body = _starter_body(display_name)
    return _write_native_skill(manifest, body, output_root=output_root)


def validate_skill(path: str | Path, *, known_tools: set[str]) -> SkillValidationReport:
    """Validate the complete native package and all locally enforceable references."""
    source = skill_file(path)
    if source.name != "SKILL.md":
        raise SkillValidationError(f"Native Skill file must be named SKILL.md: {source}")
    metadata = load_metadata(source, SkillSource.GLOBAL)
    bundle = load_bundle(metadata)
    unknown_tools = bundle.manifest.requested_tools - known_tools
    if unknown_tools:
        raise SkillValidationError(
            "Skill requests unknown tools: " + ", ".join(sorted(unknown_tools))
        )
    return SkillValidationReport(
        skill_id=bundle.manifest.id,
        version=bundle.manifest.version,
        path=source,
        estimated_prompt_tokens=bundle.estimated_prompt_tokens,
        requested_tools=tuple(sorted(bundle.manifest.requested_tools)),
    )


def convert_skill(
    source: str | Path,
    *,
    output_root: Path,
    known_tools: set[str],
    skill_id: str | None = None,
    name: str | None = None,
    summary: str | None = None,
) -> SkillConversionReport:
    """Convert supported Markdown Skill metadata into the non-executable native schema."""
    source_path = skill_file(source)
    metadata, body = _read_foreign_skill(source_path)
    source_format = _detect_format(metadata)
    resolved_id = skill_id or _source_id(metadata, source_path)
    resolved_name = name or _source_name(metadata, resolved_id)
    resolved_summary = summary or _source_summary(metadata, resolved_name)
    mapped_tools, tool_warnings = _map_tools(metadata, known_tools)
    before_edit, before_finish, requirement_warnings = _map_requirements(metadata)
    compatible_modes, mode_warnings = _map_modes(metadata)
    dropped_fields = tuple(sorted(set(metadata) - MAPPED_SOURCE_FIELDS))
    warnings = [*tool_warnings, *requirement_warnings, *mode_warnings]
    activation = metadata.get("activation")
    if isinstance(activation, dict) and activation.get("automatic"):
        warnings.append("Automatic activation was disabled; first-version routing is explicit")
    if isinstance(activation, dict) and activation.get("explicit") is False:
        warnings.append("Explicit activation was enabled for the converted Skill")
    if metadata.get("verification_profiles"):
        warnings.append(
            "Verification profiles were omitted; deterministic profile conversion is not "
            "supported yet"
        )
    if metadata.get("limits"):
        warnings.append("Source limits were reset to RepoRivet conversion defaults")
    executable = _find_executable_fields(metadata)
    if executable:
        warnings.append(
            "Executable fields were removed and never converted: " + ", ".join(executable)
        )
    unsupported = sorted(
        field
        for field in dropped_fields
        if field not in {item.split(".", 1)[0] for item in executable}
    )
    if unsupported:
        warnings.append("Unsupported metadata was omitted: " + ", ".join(unsupported))
    safe_version = _safe_version(metadata.get("version"))
    if metadata.get("version") is not None and str(metadata["version"]).strip() != safe_version:
        warnings.append(f"Unsupported version was reset to {safe_version}")

    manifest = SkillManifest.model_validate(
        {
            "schema_version": 1,
            "id": resolved_id,
            "name": resolved_name,
            "version": safe_version,
            "summary": resolved_summary,
            "category": _safe_category(metadata.get("category")),
            "activation": {"explicit": True, "automatic": False},
            "triggers": _safe_triggers(metadata.get("triggers")),
            "compatible_modes": compatible_modes,
            "requested_tools": mapped_tools,
            "requirements": {
                "before_edit": before_edit,
                "before_finish": before_finish,
            },
            # Profiles cannot be preserved until RepoRivet has a deterministic profile registry.
            "verification_profiles": [],
            "limits": {"max_prompt_tokens": 2000, "max_active_support_skills": 0},
        }
    )
    target = _write_native_skill(
        manifest,
        body or _starter_body(resolved_name),
        output_root=output_root,
    )
    validate_skill(target, known_tools=known_tools)
    return SkillConversionReport(
        source=source_path,
        target=target,
        source_format=source_format,
        mapped_tools=tuple(sorted(mapped_tools)),
        dropped_fields=dropped_fields,
        warnings=tuple(warnings),
    )


def install_skill(
    source: str | Path,
    *,
    global_root: Path,
    known_tools: set[str],
    replace: bool = False,
    reserved_ids: set[str] | frozenset[str] = frozenset(),
) -> Path:
    """Install or explicitly replace a validated global Skill."""
    report = validate_skill(source, known_tools=known_tools)
    if report.skill_id in reserved_ids:
        raise SkillValidationError(
            f"Global Skill ID {report.skill_id!r} is reserved by a system Skill"
        )
    destination = global_root.expanduser().resolve() / report.skill_id
    if destination.exists():
        if destination.is_symlink():
            raise SkillValidationError(
                f"Global Skill directory must not be a symlink: {destination}"
            )
        if not replace:
            raise SkillValidationError(
                f"Global Skill already exists: {destination}. Pass --replace to update it."
            )
        current = load_metadata(destination / "SKILL.md", SkillSource.GLOBAL)
        if current.manifest.id != report.skill_id:
            raise SkillValidationError(
                f"Global Skill directory has an unexpected ID: {destination}"
            )
        target = destination / "SKILL.md"
        atomic_write_text(target, report.path.read_text(encoding="utf-8"))
        validate_skill(target, known_tools=known_tools)
        return target
    destination.mkdir(parents=True, exist_ok=False)
    try:
        target = destination / "SKILL.md"
        atomic_write_text(target, report.path.read_text(encoding="utf-8"))
        validate_skill(target, known_tools=known_tools)
        return target
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def uninstall_skill(skill_id: str, *, global_root: Path) -> Path:
    """Remove one exact user-installed global Skill directory."""
    root = global_root.expanduser().resolve()
    unresolved = root / skill_id
    if unresolved.is_symlink():
        raise SkillValidationError(f"Global Skill directory must not be a symlink: {unresolved}")
    destination = unresolved.resolve()
    if destination.parent != root:
        raise SkillValidationError(f"Invalid global Skill ID: {skill_id!r}")
    source = destination / "SKILL.md"
    if not source.is_file():
        raise SkillValidationError(f"Unknown global Skill: {skill_id}")
    metadata = load_metadata(source, SkillSource.GLOBAL)
    if metadata.manifest.id != skill_id:
        raise SkillValidationError(f"Global Skill directory has an unexpected ID: {destination}")
    shutil.rmtree(destination)
    return destination


def _write_native_skill(manifest: SkillManifest, body: str, *, output_root: Path) -> Path:
    target = output_root.expanduser().resolve() / manifest.id / "SKILL.md"
    if target.exists():
        raise SkillValidationError(f"Refusing to overwrite existing Skill: {target}")
    estimated = ApproximateTokenEstimator(safety_factor=1.0).estimate_text(body)
    if estimated > manifest.limits.max_prompt_tokens:
        raise SkillValidationError(
            f"Skill body exceeds max_prompt_tokens ({estimated} > "
            f"{manifest.limits.max_prompt_tokens})"
        )
    payload = manifest.model_dump(mode="json")
    payload["compatible_modes"] = sorted(manifest.compatible_modes)
    payload["requested_tools"] = sorted(manifest.requested_tools)
    yaml_text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    ).strip()
    atomic_write_text(target, f"---\n{yaml_text}\n---\n\n{body.strip()}\n")
    return target


def _read_foreign_skill(path: Path) -> tuple[dict[str, Any], str]:
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
    if metadata.get("schema_version") == 1 and "requested_tools" in metadata:
        return "repo-rivet"
    if "allowed-tools" in metadata or "allowed_tools" in metadata:
        return "claude"
    if "description" in metadata and "name" in metadata:
        return "codex"
    return "generic-markdown"


def _source_id(metadata: dict[str, Any], source: Path) -> str:
    candidate = str(metadata.get("id") or metadata.get("name") or source.parent.name)
    normalized = re.sub(r"[^a-z0-9]+", "-", candidate.strip().lower()).strip("-")
    if len(normalized) < 2:
        raise SkillValidationError("Could not derive a valid Skill ID; pass --id explicitly")
    return normalized[:64].rstrip("-")


def _source_name(metadata: dict[str, Any], skill_id: str) -> str:
    value = str(metadata.get("name") or "").strip()
    return value[:100] if value else _title_from_id(skill_id)


def _source_summary(metadata: dict[str, Any], name: str) -> str:
    value = str(metadata.get("summary") or metadata.get("description") or "").strip()
    return value[:500] if value else f"Converted workflow for {name} tasks."


def _safe_version(value: Any) -> str:
    rendered = str(value or "1.0.0").strip()
    return rendered if re.fullmatch(r"\d+\.\d+\.\d+", rendered) else "1.0.0"


def _safe_category(value: Any) -> str:
    rendered = str(value or "workflow").strip()
    return rendered[:50] or "workflow"


def _safe_triggers(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[str]] = {}
    for key in ("task_types", "file_globs", "project_markers", "keywords"):
        items = value.get(key)
        if isinstance(items, list):
            result[key] = [str(item)[:200] for item in items[:50]]
    return result


def _map_tools(metadata: dict[str, Any], known_tools: set[str]) -> tuple[list[str], list[str]]:
    raw = (
        metadata.get("requested_tools")
        or metadata.get("allowed-tools")
        or metadata.get("allowed_tools")
        or metadata.get("tools")
    )
    values = _string_list(raw)
    mapped: set[str] = set()
    unknown: list[str] = []
    for value in values:
        base = re.split(r"[\s(:]", value, maxsplit=1)[0]
        normalized = re.sub(r"[^a-z0-9_]", "", base.lower())
        target = value if value in known_tools else FOREIGN_TOOL_ALIASES.get(normalized)
        if target in known_tools:
            mapped.add(target)
        else:
            unknown.append(value)
    warnings: list[str] = []
    if unknown:
        warnings.append("Unmapped tools were omitted: " + ", ".join(sorted(set(unknown))))
    if not mapped:
        mapped.update({"list_files", "read_file", "search_text"} & known_tools)
        warnings.append("No compatible tools were declared; applied the read-only default set")
    return sorted(mapped), warnings


def _map_requirements(
    metadata: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    value = metadata.get("requirements")
    if not isinstance(value, dict):
        return [], [], []
    before_edit = _string_list(value.get("before_edit"))
    before_finish = _string_list(value.get("before_finish"))
    unknown = (set(before_edit) | set(before_finish)) - FIXED_REQUIREMENTS
    warnings = (
        ["Unknown completion requirements were omitted: " + ", ".join(sorted(unknown))]
        if unknown
        else []
    )
    return (
        [item for item in before_edit if item in FIXED_REQUIREMENTS],
        [item for item in before_finish if item in FIXED_REQUIREMENTS],
        warnings,
    )


def _map_modes(metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    values = _string_list(metadata.get("compatible_modes"))
    if not values:
        return ["plan", "execute"], []
    supported = sorted(set(values) & {"plan", "execute"})
    unknown = sorted(set(values) - {"plan", "execute"})
    warnings = ["Unsupported modes were omitted: " + ", ".join(unknown)] if unknown else []
    if not supported:
        supported = ["plan", "execute"]
        warnings.append("No compatible mode remained; applied plan and execute")
    return supported, warnings


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        return [item.strip() for item in re.findall(r"\S+\([^)]*\)|\S+", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


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


def _title_from_id(skill_id: str) -> str:
    return " ".join(part.capitalize() for part in skill_id.split("-") if part)


def _starter_body(name: str) -> str:
    return f"""# Objective

Describe the concrete outcome this Skill helps achieve for {name} tasks.

# Procedure

1. Inspect the smallest relevant set of files and collect direct evidence.
2. State the bounded intended change or result.
3. Use only requested tools and respect the current RepoRivet mode.
4. Verify every locally enforceable completion requirement.

# Constraints

- Do not claim permissions that RepoRivet has not granted.
- Do not bypass approval, workspace policy, snapshots, or verification.
- Keep instructions declarative; do not embed executable hooks or dynamic code.

# Completion Conditions

Describe the observable evidence required before the task can be reported complete."""
