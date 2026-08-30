"""Typed snapshots, visible ranges, line operations, and edit results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_EDIT_OPERATIONS = 10
MAX_NEW_LINES_PER_OPERATION = 500
RECOVERY_MAX_EDIT_OPERATIONS = 1
RECOVERY_MAX_NEW_LINES = 40


class EditError(ValueError):
    """Expected edit rejection with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.metadata = metadata


class FileSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    display_tag: str
    relative_path: str
    normalized_content_hash: str
    raw_bytes_hash: str
    encoding: Literal["utf-8", "utf-8-sig"]
    newline_style: Literal["lf", "crlf"]
    has_trailing_newline: bool
    total_lines: int = Field(ge=0)
    normalized_content: str = Field(repr=False)
    parent_snapshot_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VisibleRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    snapshot_id: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    source: Literal["read_file", "search_text", "edit_file"]


class _NewLinesOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_lines: list[str] = Field(
        default_factory=list,
        max_length=MAX_NEW_LINES_PER_OPERATION,
    )

    @model_validator(mode="after")
    def validate_lines(self) -> _NewLinesOperation:
        if any("\n" in line or "\r" in line for line in self.new_lines):
            raise ValueError("new_lines entries must not contain newline characters")
        return self


class ReplaceLines(_NewLinesOperation):
    op: Literal["replace"]
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> ReplaceLines:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class InsertBefore(_NewLinesOperation):
    op: Literal["insert_before"]
    line: int = Field(ge=1)


class InsertAfter(_NewLinesOperation):
    op: Literal["insert_after"]
    line: int = Field(ge=1)


class InsertStart(_NewLinesOperation):
    op: Literal["insert_start"]


class InsertEnd(_NewLinesOperation):
    op: Literal["insert_end"]


class DeleteLines(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["delete"]
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> DeleteLines:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


EditOperation = Annotated[
    ReplaceLines | InsertBefore | InsertAfter | InsertStart | InsertEnd | DeleteLines,
    Field(discriminator="op"),
]


class EditFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    snapshot_id: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    operations: list[EditOperation] = Field(min_length=1, max_length=MAX_EDIT_OPERATIONS)


class TextSplice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    replacement_lines: list[str]
    operation_index: int = Field(ge=0)


class EditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    old_snapshot_id: str
    old_snapshot_tag: str
    new_snapshot_id: str
    new_snapshot_tag: str
    changed_ranges: list[tuple[int, int]]
    bytes_before: int = Field(ge=0)
    bytes_after: int = Field(ge=0)
    workspace_revision: int = Field(ge=1)
    diff_preview: str
    recovered_from_stale_snapshot: bool = False
