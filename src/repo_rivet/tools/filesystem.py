"""Workspace-confined file listing, search, read, edit, and deletion tools."""

import hashlib
import json
import re
import shutil
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, model_validator

from repo_rivet.approval.models import Capability
from repo_rivet.editing.atomic_writer import atomic_create_bytes
from repo_rivet.editing.document import MAX_TEXT_FILE_BYTES, TextDocument
from repo_rivet.editing.runtime import EditingRuntime
from repo_rivet.safety.path_policy import PathPolicyError, WorkspacePathPolicy
from repo_rivet.tools.base import BaseTool, DecisionPolicy, ToolArguments, ToolResult

MAX_READ_LINES = 300
MAX_READ_CHARS = 20_000
MAX_SEARCH_MATCHES = 200
MAX_LIST_ENTRIES = 1_000
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
)
_SENSITIVE_FILE_NAMES = frozenset({".npmrc", ".pypirc", "reporivet.toml"})


class ListFilesArguments(ToolArguments):
    path: str = "."
    max_depth: int = Field(default=2, ge=1, le=10)


class SearchTextArguments(ToolArguments):
    query: str = Field(min_length=1)
    path: str = "."
    regex: bool = False
    case_sensitive: bool = True


class ReadFileArguments(ToolArguments):
    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_line_range(self) -> "ReadFileArguments":
        if self.end_line is not None:
            if self.end_line < self.start_line:
                raise ValueError("end_line must be greater than or equal to start_line")
            if self.end_line - self.start_line + 1 > MAX_READ_LINES:
                raise ValueError(f"cannot read more than {MAX_READ_LINES} lines at once")
        return self


class WriteFileArguments(ToolArguments):
    path: str
    content: str


class DeletePathArguments(ToolArguments):
    path: str
    recursive: bool = Field(
        default=False,
        description="Must be true to delete a non-empty directory.",
    )


@dataclass(frozen=True, slots=True)
class PreparedDeletion:
    key: str
    path: Path
    relative_path: str
    entry_type: str
    entry_count: int
    total_bytes: int
    fingerprint: str
    protected_entries: tuple[str, ...]


class WorkspaceTool[ArgumentsT: ToolArguments](BaseTool[ArgumentsT]):
    """Base class for tools sharing one workspace path policy."""

    arguments_type: ClassVar[type[ToolArguments]] = ToolArguments

    def __init__(self, path_policy: WorkspacePathPolicy) -> None:
        self.path_policy = path_policy


class ListFilesTool(WorkspaceTool[ListFilesArguments]):
    name = "list_files"
    description = "List files and directories inside the workspace to a limited depth."
    arguments_type = ListFilesArguments
    capabilities = frozenset({Capability.FILESYSTEM_READ})

    def run(self, arguments: ListFilesArguments) -> ToolResult:
        root = self.path_policy.resolve(arguments.path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {arguments.path}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {arguments.path}")

        entries: list[str] = []
        truncated = self._walk(root, root, arguments.max_depth, entries)
        return ToolResult(
            ok=True,
            output="\n".join(entries) if entries else "(no files)",
            metadata={"entries": len(entries), "truncated": truncated},
        )

    def _walk(self, root: Path, directory: Path, max_depth: int, entries: list[str]) -> bool:
        depth = len(directory.relative_to(root).parts)
        if depth >= max_depth:
            return False

        try:
            children = sorted(
                directory.iterdir(), key=lambda child: (not child.is_dir(), child.name)
            )
        except OSError as error:
            raise ValueError(f"Could not list directory: {directory.name}: {error}") from error

        for child in children:
            if len(entries) >= MAX_LIST_ENTRIES:
                return True
            relative = child.relative_to(self.path_policy.workspace).as_posix()
            if child.is_symlink():
                entries.append(f"{relative}@")
                continue
            if child.is_dir():
                if child.name in _EXCLUDED_DIRECTORY_NAMES:
                    continue
                entries.append(f"{relative}/")
                if self._walk(root, child, max_depth, entries):
                    return True
            else:
                entries.append(relative)
        return False


class SearchTextTool(WorkspaceTool[SearchTextArguments]):
    name = "search_text"
    description = "Search text files in the workspace using a literal string or regular expression."
    arguments_type = SearchTextArguments
    capabilities = frozenset({Capability.FILESYSTEM_READ})

    def __init__(
        self,
        path_policy: WorkspacePathPolicy,
        editing_runtime: EditingRuntime | None = None,
    ) -> None:
        super().__init__(path_policy)
        self.editing_runtime = editing_runtime or EditingRuntime(path_policy)

    def run(self, arguments: SearchTextArguments) -> ToolResult:
        root = self.path_policy.resolve(arguments.path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {arguments.path}")

        flags = 0 if arguments.case_sensitive else re.IGNORECASE
        expression = arguments.query if arguments.regex else re.escape(arguments.query)
        try:
            pattern = re.compile(expression, flags)
        except re.error as error:
            raise ValueError(f"Invalid regular expression: {error}") from None

        matches: list[str] = []
        match_locations: list[str] = []
        snapshot_ids: dict[str, str] = {}
        files_scanned = 0
        for display_path, file_path in self._iter_files(root):
            try:
                document = TextDocument.load(file_path)
            except (OSError, UnicodeError, ValueError):
                continue
            files_scanned += 1
            snapshot = None
            for line_number, line in enumerate(document.lines, start=1):
                if pattern.search(line):
                    if snapshot is None:
                        snapshot = self.editing_runtime.capture_document(
                            display_path,
                            document,
                            start_line=line_number,
                            end_line=line_number,
                            source="search_text",
                            visible=len(line) <= 500,
                        )
                        snapshot_ids[display_path] = snapshot.snapshot_id
                    elif len(line) <= 500:
                        self.editing_runtime.visibility.record(
                            path=display_path,
                            snapshot_id=snapshot.snapshot_id,
                            start_line=line_number,
                            end_line=line_number,
                            source="search_text",
                        )
                    matches.append(f"{display_path}:{line_number}:{line[:500]}")
                    if len(match_locations) < 5:
                        match_locations.append(f"{display_path}:{line_number}")
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        return ToolResult(
                            ok=True,
                            output="\n".join(matches),
                            metadata={
                                "matches": len(matches),
                                "match_locations": match_locations,
                                "files_scanned": files_scanned,
                                "truncated": True,
                                "snapshot_ids": snapshot_ids,
                            },
                        )

        return ToolResult(
            ok=True,
            output="\n".join(matches) if matches else "(no matches)",
            metadata={
                "matches": len(matches),
                "match_locations": match_locations,
                "files_scanned": files_scanned,
                "truncated": False,
                "snapshot_ids": snapshot_ids,
            },
        )

    def _iter_files(self, root: Path) -> Iterator[tuple[str, Path]]:
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for candidate in candidates:
            try:
                relative = candidate.relative_to(self.path_policy.workspace)
            except ValueError:
                continue
            if any(part in _EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
                continue
            if _is_sensitive_path(relative.as_posix()):
                continue
            try:
                resolved = self.path_policy.resolve(relative)
            except PathPolicyError:
                continue
            if resolved.is_file():
                yield relative.as_posix(), resolved


class ReadFileTool(WorkspaceTool[ReadFileArguments]):
    name = "read_file"
    description = "Read up to 300 numbered lines from a UTF-8 text file in the workspace."
    arguments_type = ReadFileArguments
    capabilities = frozenset({Capability.FILESYSTEM_READ})

    def __init__(
        self,
        path_policy: WorkspacePathPolicy,
        editing_runtime: EditingRuntime | None = None,
    ) -> None:
        super().__init__(path_policy)
        self.editing_runtime = editing_runtime or EditingRuntime(path_policy)

    def run(self, arguments: ReadFileArguments) -> ToolResult:
        if _is_sensitive_path(arguments.path):
            raise ValueError(f"Sensitive configuration files cannot be read: {arguments.path}")
        file_path = self.path_policy.resolve(arguments.path)
        document = TextDocument.load(file_path)
        lines = document.lines
        if arguments.start_line > max(len(lines), 1):
            raise ValueError(f"start_line exceeds file length ({len(lines)} lines)")

        requested_end = arguments.end_line or arguments.start_line + MAX_READ_LINES - 1
        actual_end = min(requested_end, len(lines))
        selected = lines[arguments.start_line - 1 : actual_end]
        rendered: list[str] = []
        rendered_chars = 0
        complete_lines = 0
        for line_number, line in enumerate(selected, start=arguments.start_line):
            item = f"{line_number}│ {line}"
            if rendered_chars + len(item) + 1 > MAX_READ_CHARS:
                if not rendered:
                    suffix = " ... line truncated; read a narrower source representation ..."
                    rendered.append(item[: MAX_READ_CHARS - len(suffix)] + suffix)
                break
            rendered.append(item)
            rendered_chars += len(item) + 1
            complete_lines += 1
        chars_truncated = complete_lines < len(selected)
        displayed_end = arguments.start_line + complete_lines - 1
        snapshot = self.editing_runtime.capture_document(
            arguments.path,
            document,
            start_line=arguments.start_line,
            end_line=max(arguments.start_line, displayed_end),
            source="read_file",
            visible=complete_lines > 0 or not lines,
        )
        if complete_lines:
            shown = f"{arguments.start_line}-{displayed_end}"
        elif not lines:
            shown = "empty"
        else:
            shown = f"none partial={arguments.start_line}"
        header = (
            f"[{snapshot.relative_path}#{snapshot.display_tag} lines={shown} total={len(lines)}]"
        )
        output = "\n".join([header, *rendered])
        if chars_truncated:
            output += "\n... output truncated by character limit ..."
        return ToolResult(
            ok=True,
            output=output,
            metadata={
                "path": snapshot.relative_path,
                "sha256": document.raw_hash,
                "raw_bytes_hash": document.raw_hash,
                "normalized_content_hash": document.normalized_hash,
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_tag": snapshot.display_tag,
                "total_lines": len(lines),
                "start_line": arguments.start_line,
                "end_line": max(arguments.start_line, displayed_end),
                "fully_visible_end_line": displayed_end if complete_lines else None,
                "encoding": snapshot.encoding,
                "newline_style": snapshot.newline_style,
                "has_trailing_newline": snapshot.has_trailing_newline,
                "truncated": displayed_end < len(lines) or chars_truncated,
            },
            raw_output=output,
        )


class WriteFileTool(WorkspaceTool[WriteFileArguments]):
    name = "write_file"
    description = "Create a new UTF-8 text file. Existing files must be changed with edit_file."
    arguments_type = WriteFileArguments
    capabilities = frozenset({Capability.FILESYSTEM_WRITE})

    def __init__(
        self,
        path_policy: WorkspacePathPolicy,
        editing_runtime: EditingRuntime | None = None,
    ) -> None:
        super().__init__(path_policy)
        self.editing_runtime = editing_runtime or EditingRuntime(path_policy)

    def run(self, arguments: WriteFileArguments) -> ToolResult:
        if _is_sensitive_path(arguments.path):
            raise ValueError(f"Sensitive configuration files cannot be written: {arguments.path}")
        encoded_content = arguments.content.encode("utf-8")
        if len(encoded_content) > MAX_TEXT_FILE_BYTES:
            raise ValueError(f"Content exceeds {MAX_TEXT_FILE_BYTES} bytes")
        desired_document = TextDocument.from_bytes(encoded_content)

        file_path = self.path_policy.resolve(arguments.path)
        if file_path.exists() and file_path.is_dir():
            raise ValueError(f"Path is a directory: {arguments.path}")
        if file_path.exists():
            raise ValueError("File already exists; use read_file and edit_file to change it")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            atomic_create_bytes(file_path, encoded_content)
        except FileExistsError:
            raise ValueError("File was created concurrently; no content was overwritten") from None
        document, snapshot = self.editing_runtime.capture(
            arguments.path,
            start_line=1,
            end_line=max(1, desired_document.total_lines),
            source="edit_file",
        )
        workspace_revision = self.editing_runtime.record_created_file()
        line_count = document.total_lines
        return ToolResult(
            ok=True,
            output=f"Wrote {len(encoded_content)} bytes to {arguments.path}",
            metadata={
                "path": snapshot.relative_path,
                "bytes": len(encoded_content),
                "line_count": line_count,
                "created": True,
                "sha256": document.raw_hash,
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_tag": snapshot.display_tag,
                "workspace_revision": workspace_revision,
            },
            raw_output=arguments.content,
        )


class DeletePathTool(WorkspaceTool[DeletePathArguments]):
    name = "delete_path"
    description = (
        "Delete one workspace file, symlink, or directory. Set recursive=true explicitly for "
        "a non-empty directory. The workspace root and protected repository metadata cannot "
        "be deleted."
    )
    arguments_type = DeletePathArguments
    capabilities = frozenset({Capability.FILESYSTEM_DELETE})
    decision_policy = DecisionPolicy.APPROVAL_GATED

    def __init__(
        self,
        path_policy: WorkspacePathPolicy,
        editing_runtime: EditingRuntime | None = None,
    ) -> None:
        super().__init__(path_policy)
        self.editing_runtime = editing_runtime or EditingRuntime(path_policy)
        self._prepared: dict[str, PreparedDeletion] = {}

    def approval_arguments(self, arguments: DeletePathArguments) -> dict[str, Any] | ToolResult:
        try:
            prepared = self._prepare(arguments)
        except (OSError, ValueError) as error:
            return ToolResult(
                ok=False,
                output="",
                error=str(error),
                error_code="invalid_delete_target",
                retryable=False,
            )
        return {
            "path": prepared.relative_path,
            "recursive": arguments.recursive,
            "entry_type": prepared.entry_type,
            "entry_count": prepared.entry_count,
            "total_bytes": prepared.total_bytes,
            "_prepared_fingerprint": prepared.fingerprint,
        }

    def run(self, arguments: DeletePathArguments) -> ToolResult:
        prepared = self._prepare(arguments)
        current = self._inspect(prepared.path, prepared.key)
        if current.fingerprint != prepared.fingerprint:
            self._prepared.pop(prepared.key, None)
            raise ValueError(
                "Deletion target changed during approval; inspect it again before deleting"
            )

        if prepared.entry_type == "directory":
            if arguments.recursive:
                shutil.rmtree(prepared.path)
            else:
                prepared.path.rmdir()
        else:
            prepared.path.unlink()
        self._prepared.pop(prepared.key, None)
        workspace_revision = self.editing_runtime.record_deleted_path()
        return ToolResult(
            ok=True,
            output=f"Deleted {prepared.entry_type} {prepared.relative_path}",
            metadata={
                "path": prepared.relative_path,
                "deleted": True,
                "path_type": prepared.entry_type,
                "entry_count": prepared.entry_count,
                "bytes": prepared.total_bytes,
                "workspace_revision": workspace_revision,
            },
        )

    def _prepare(self, arguments: DeletePathArguments) -> PreparedDeletion:
        key = json.dumps(arguments.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        cached = self._prepared.get(key)
        if cached is not None:
            return cached
        path = self.path_policy.resolve_entry(arguments.path)
        if path == self.path_policy.workspace:
            raise ValueError("The workspace root cannot be deleted")
        relative = path.relative_to(self.path_policy.workspace).as_posix()
        if relative == ".git" or relative.startswith(".git/"):
            raise ValueError("Git repository metadata cannot be deleted")
        if _is_sensitive_path(relative):
            raise ValueError(f"Sensitive configuration files cannot be deleted: {relative}")
        prepared = self._inspect(path, key)
        if prepared.protected_entries:
            preview = ", ".join(prepared.protected_entries[:3])
            raise ValueError(
                f"Protected repository or configuration entries cannot be deleted: {preview}"
            )
        if (
            prepared.entry_type == "directory"
            and prepared.entry_count > 0
            and not arguments.recursive
        ):
            raise ValueError("Directory is not empty; set recursive=true to delete its contents")
        self._prepared[key] = prepared
        return prepared

    def _inspect(self, path: Path, key: str) -> PreparedDeletion:
        try:
            root_stat = path.lstat()
        except FileNotFoundError:
            raise ValueError(f"Path does not exist: {path.name}") from None

        if stat.S_ISLNK(root_stat.st_mode):
            entry_type = "symlink"
            entries = [(".", root_stat)]
        elif stat.S_ISREG(root_stat.st_mode):
            entry_type = "file"
            entries = [(".", root_stat)]
        elif stat.S_ISDIR(root_stat.st_mode):
            entry_type = "directory"
            entries = []
            pending = [path]
            while pending:
                directory = pending.pop()
                for child in sorted(directory.iterdir(), key=lambda item: item.name):
                    child_stat = child.lstat()
                    entries.append((child.relative_to(path).as_posix(), child_stat))
                    if stat.S_ISDIR(child_stat.st_mode) and not stat.S_ISLNK(child_stat.st_mode):
                        pending.append(child)
        else:
            raise ValueError("Only regular files, symlinks, and directories can be deleted")

        digest = hashlib.sha256()
        total_bytes = 0
        for relative, entry_stat in sorted(entries, key=lambda item: item[0]):
            total_bytes += entry_stat.st_size if stat.S_ISREG(entry_stat.st_mode) else 0
            digest.update(
                json.dumps(
                    [
                        relative,
                        stat.S_IFMT(entry_stat.st_mode),
                        entry_stat.st_size,
                        entry_stat.st_mtime_ns,
                        entry_stat.st_ino,
                    ],
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        relative_path = path.relative_to(self.path_policy.workspace).as_posix()
        protected_entries: list[str] = []
        for relative, _ in entries:
            full_relative = (
                relative_path if relative == "." else (Path(relative_path) / relative).as_posix()
            )
            if ".git" in Path(full_relative).parts or _is_sensitive_path(full_relative):
                protected_entries.append(full_relative)
        return PreparedDeletion(
            key=key,
            path=path,
            relative_path=relative_path,
            entry_type=entry_type,
            entry_count=(len(entries) if entry_type == "directory" else 1),
            total_bytes=total_bytes,
            fingerprint=digest.hexdigest(),
            protected_entries=tuple(protected_entries),
        )


def _is_sensitive_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name in _SENSITIVE_FILE_NAMES or name == ".env" or name.startswith(".env.")
