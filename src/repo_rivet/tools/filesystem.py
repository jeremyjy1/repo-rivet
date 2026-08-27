"""Workspace-confined file listing, search, read, and edit tools."""

import re
from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator

from repo_rivet.safety.path_policy import PathPolicyError, WorkspacePathPolicy
from repo_rivet.tools.base import BaseTool, ToolArguments, ToolResult

MAX_TEXT_FILE_BYTES = 1_000_000
MAX_READ_LINES = 300
MAX_SEARCH_MATCHES = 200
MAX_LIST_ENTRIES = 1_000
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
)


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
    overwrite: bool = False


class ReplaceTextArguments(ToolArguments):
    path: str
    old_text: str = Field(min_length=1)
    new_text: str
    expected_count: int = Field(default=1, ge=1)


class WorkspaceTool(BaseTool[ToolArguments]):
    """Base class for tools sharing one workspace path policy."""

    arguments_type: ClassVar[type[ToolArguments]] = ToolArguments

    def __init__(self, path_policy: WorkspacePathPolicy) -> None:
        self.path_policy = path_policy


class ListFilesTool(WorkspaceTool):
    name = "list_files"
    description = "List files and directories inside the workspace to a limited depth."
    arguments_type = ListFilesArguments

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
        files_scanned = 0
        for display_path, file_path in self._iter_files(root):
            try:
                content = _read_text(file_path)
            except (OSError, UnicodeError, ValueError):
                continue
            files_scanned += 1
            for line_number, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(f"{display_path}:{line_number}:{line[:500]}")
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        return ToolResult(
                            ok=True,
                            output="\n".join(matches),
                            metadata={
                                "matches": len(matches),
                                "files_scanned": files_scanned,
                                "truncated": True,
                            },
                        )

        return ToolResult(
            ok=True,
            output="\n".join(matches) if matches else "(no matches)",
            metadata={
                "matches": len(matches),
                "files_scanned": files_scanned,
                "truncated": False,
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

    def run(self, arguments: ReadFileArguments) -> ToolResult:
        file_path = self.path_policy.resolve(arguments.path)
        content = _read_text(file_path)
        lines = content.splitlines()
        if arguments.start_line > max(len(lines), 1):
            raise ValueError(f"start_line exceeds file length ({len(lines)} lines)")

        requested_end = arguments.end_line or arguments.start_line + MAX_READ_LINES - 1
        actual_end = min(requested_end, len(lines))
        selected = lines[arguments.start_line - 1 : actual_end]
        output = "\n".join(
            f"{line_number:>6} | {line}"
            for line_number, line in enumerate(selected, start=arguments.start_line)
        )
        return ToolResult(
            ok=True,
            output=output,
            metadata={
                "total_lines": len(lines),
                "start_line": arguments.start_line,
                "end_line": actual_end,
                "truncated": actual_end < len(lines),
            },
        )


class WriteFileTool(WorkspaceTool):
    name = "write_file"
    description = (
        "Create a UTF-8 text file, or overwrite one only when overwrite is explicitly true."
    )
    arguments_type = WriteFileArguments

    def run(self, arguments: WriteFileArguments) -> ToolResult:
        encoded_content = arguments.content.encode("utf-8")
        if len(encoded_content) > MAX_TEXT_FILE_BYTES:
            raise ValueError(f"Content exceeds {MAX_TEXT_FILE_BYTES} bytes")

        file_path = self.path_policy.resolve(arguments.path)
        existed = file_path.exists()
        if existed and file_path.is_dir():
            raise ValueError(f"Path is a directory: {arguments.path}")
        if existed and not arguments.overwrite:
            raise ValueError("File already exists; set overwrite=true to replace it")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(arguments.content, encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"Wrote {len(encoded_content)} bytes to {arguments.path}",
            metadata={
                "path": arguments.path,
                "bytes": len(encoded_content),
                "created": not existed,
            },
        )


class ReplaceTextTool(WorkspaceTool):
    name = "replace_text"
    description = "Replace exact text only when its occurrence count matches expected_count."
    arguments_type = ReplaceTextArguments

    def run(self, arguments: ReplaceTextArguments) -> ToolResult:
        file_path = self.path_policy.resolve(arguments.path)
        content = _read_text(file_path)
        actual_count = content.count(arguments.old_text)
        if actual_count != arguments.expected_count:
            raise ValueError(
                f"Expected {arguments.expected_count} matches but found {actual_count}; "
                "file unchanged"
            )

        updated = content.replace(arguments.old_text, arguments.new_text)
        encoded_content = updated.encode("utf-8")
        if len(encoded_content) > MAX_TEXT_FILE_BYTES:
            raise ValueError(f"Updated content exceeds {MAX_TEXT_FILE_BYTES} bytes")
        file_path.write_text(updated, encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"Replaced {actual_count} occurrence(s) in {arguments.path}",
            metadata={"path": arguments.path, "replacements": actual_count},
        )


def _read_text(file_path: Path) -> str:
    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path.name}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path.name}")
    if file_path.stat().st_size > MAX_TEXT_FILE_BYTES:
        raise ValueError(f"File exceeds {MAX_TEXT_FILE_BYTES} bytes: {file_path.name}")

    data = file_path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"Binary files are not supported: {file_path.name}")
    return data.decode("utf-8")
