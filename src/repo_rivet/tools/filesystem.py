"""Workspace-confined file listing, search, read, and edit tools."""

import re
from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator

from repo_rivet.approval.models import Capability
from repo_rivet.editing.atomic_writer import atomic_create_bytes
from repo_rivet.editing.document import MAX_TEXT_FILE_BYTES, TextDocument
from repo_rivet.editing.runtime import EditingRuntime
from repo_rivet.safety.path_policy import PathPolicyError, WorkspacePathPolicy
from repo_rivet.tools.base import BaseTool, ToolArguments, ToolResult

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


class WorkspaceTool(BaseTool[ToolArguments]):
    """Base class for tools sharing one workspace path policy."""

    arguments_type: ClassVar[type[ToolArguments]] = ToolArguments

    def __init__(self, path_policy: WorkspacePathPolicy) -> None:
        self.path_policy = path_policy


class ListFilesTool(WorkspaceTool):
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


class SearchTextTool(WorkspaceTool):
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

    def _iter_files(self, root: Path):  # type: ignore[no-untyped-def]
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


class ReadFileTool(WorkspaceTool):
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


class WriteFileTool(WorkspaceTool):
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


def _is_sensitive_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name in _SENSITIVE_FILE_NAMES or name == ".env" or name.startswith(".env.")
