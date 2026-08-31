"""Resolve snapshot-relative line operations into one preflighted document."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from repo_rivet.editing.document import (
    MAX_TEXT_FILE_BYTES,
    TextDocument,
    split_normalized_lines,
)
from repo_rivet.editing.models import (
    DeleteLines,
    EditError,
    EditFileArguments,
    FileSnapshot,
    InsertAfter,
    InsertBefore,
    InsertEnd,
    InsertStart,
    ReplaceLines,
    TextSplice,
)
from repo_rivet.editing.snapshot_store import VisibilityStore

MAX_DIFF_PREVIEW_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class PlannedEdit:
    desired_document: TextDocument
    diff_preview: str
    changed_ranges: list[tuple[int, int]]


def plan_edit(
    snapshot: FileSnapshot,
    arguments: EditFileArguments,
    visibility: VisibilityStore,
) -> PlannedEdit:
    base_lines = split_normalized_lines(snapshot.normalized_content)
    splices: list[TextSplice] = []
    for index, operation in enumerate(arguments.operations):
        if isinstance(operation, (ReplaceLines, DeleteLines)):
            _require_bounds(operation.start_line, operation.end_line, snapshot.total_lines)
            visibility.require(
                path=snapshot.relative_path,
                snapshot_id=snapshot.snapshot_id,
                start_line=operation.start_line,
                end_line=operation.end_line,
            )
            replacement = operation.new_lines if isinstance(operation, ReplaceLines) else []
            splices.append(
                TextSplice(
                    start_index=operation.start_line - 1,
                    end_index=operation.end_line,
                    replacement_lines=replacement,
                    operation_index=index,
                )
            )
        elif isinstance(operation, (InsertBefore, InsertAfter)):
            _require_bounds(operation.line, operation.line, snapshot.total_lines)
            visibility.require(
                path=snapshot.relative_path,
                snapshot_id=snapshot.snapshot_id,
                start_line=operation.line,
                end_line=operation.line,
            )
            insertion_index = (
                operation.line - 1 if isinstance(operation, InsertBefore) else operation.line
            )
            splices.append(
                TextSplice(
                    start_index=insertion_index,
                    end_index=insertion_index,
                    replacement_lines=operation.new_lines,
                    operation_index=index,
                )
            )
        elif isinstance(operation, (InsertStart, InsertEnd)):
            if not visibility.has_snapshot_view(
                path=snapshot.relative_path,
                snapshot_id=snapshot.snapshot_id,
            ):
                raise EditError(
                    "unseen_range",
                    "The file snapshot has not been shown; read it before inserting",
                    metadata={
                        "path": snapshot.relative_path,
                        "snapshot_id": snapshot.snapshot_id,
                        "required_start_line": 1,
                        "required_end_line": max(1, snapshot.total_lines),
                    },
                )
            if snapshot.total_lines:
                anchor = 1 if isinstance(operation, InsertStart) else snapshot.total_lines
                visibility.require(
                    path=snapshot.relative_path,
                    snapshot_id=snapshot.snapshot_id,
                    start_line=anchor,
                    end_line=anchor,
                )
            insertion_index = 0 if isinstance(operation, InsertStart) else snapshot.total_lines
            splices.append(
                TextSplice(
                    start_index=insertion_index,
                    end_index=insertion_index,
                    replacement_lines=operation.new_lines,
                    operation_index=index,
                )
            )
        else:  # pragma: no cover - discriminated schema is exhaustive
            raise EditError("invalid_operation", "Unsupported edit operation", retryable=False)

    _validate_non_overlapping(splices)
    desired_lines = list(base_lines)
    for splice in sorted(
        splices,
        key=lambda item: (
            item.start_index,
            item.start_index != item.end_index,
        ),
        reverse=True,
    ):
        desired_lines[splice.start_index : splice.end_index] = splice.replacement_lines
    desired_content = "\n".join(desired_lines)
    if desired_lines and snapshot.has_trailing_newline:
        desired_content += "\n"
    desired = TextDocument.from_normalized(
        desired_content,
        encoding=snapshot.encoding,
        newline_style=snapshot.newline_style,
    )
    if len(desired.raw_bytes) > MAX_TEXT_FILE_BYTES:
        raise EditError(
            "file_too_large",
            f"Edited file exceeds {MAX_TEXT_FILE_BYTES} bytes",
            retryable=False,
        )
    diff = _create_diff(snapshot.relative_path, snapshot.normalized_content, desired_content)
    changed = _changed_ranges(base_lines, desired_lines)
    return PlannedEdit(desired_document=desired, diff_preview=diff, changed_ranges=changed)


def _require_bounds(start_line: int, end_line: int, total_lines: int) -> None:
    if total_lines == 0 or start_line > total_lines or end_line > total_lines:
        raise EditError(
            "line_out_of_bounds",
            f"Target lines {start_line}-{end_line} exceed snapshot length ({total_lines})",
        )


def _validate_non_overlapping(splices: list[TextSplice]) -> None:
    ordered = sorted(splices, key=lambda item: (item.start_index, item.end_index))
    for index, current in enumerate(ordered):
        for other in ordered[index + 1 :]:
            if other.start_index > current.end_index:
                break
            current_insert = current.start_index == current.end_index
            other_insert = other.start_index == other.end_index
            overlaps = (
                current.start_index < other.end_index and other.start_index < current.end_index
            )
            insert_conflict = (
                current_insert
                and other_insert
                and current.start_index == other.start_index
                or current_insert
                and other.start_index < current.start_index < other.end_index
                or other_insert
                and current.start_index < other.start_index < current.end_index
            )
            if overlaps or insert_conflict:
                raise EditError(
                    "overlapping_operations",
                    f"Edit operations {current.operation_index} and "
                    f"{other.operation_index} overlap",
                )


def _create_diff(path: str, old: str, new: str) -> str:
    rendered = "".join(
        difflib.unified_diff(
            _lf_lines_keepends(old),
            _lf_lines_keepends(new),
            fromfile=path,
            tofile=path,
        )
    )
    if len(rendered) > MAX_DIFF_PREVIEW_CHARS:
        return rendered[:MAX_DIFF_PREVIEW_CHARS] + "\n... diff preview truncated ..."
    return rendered


def _lf_lines_keepends(content: str) -> list[str]:
    if not content:
        return []
    parts = content.split("\n")
    lines = [f"{part}\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _changed_ranges(old_lines: list[str], new_lines: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        start = new_start + 1
        end = max(start, new_end)
        ranges.append((start, end))
    return ranges
