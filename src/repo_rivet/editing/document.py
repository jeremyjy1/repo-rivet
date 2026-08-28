"""UTF-8 text loading with stable normalized line semantics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from repo_rivet.editing.models import EditError, FileSnapshot

UTF8_BOM = b"\xef\xbb\xbf"
MAX_TEXT_FILE_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class TextDocument:
    raw_bytes: bytes
    normalized_content: str
    encoding: Literal["utf-8", "utf-8-sig"]
    newline_style: Literal["lf", "crlf"]
    has_trailing_newline: bool

    @classmethod
    def load(cls, path: Path) -> TextDocument:
        if not path.exists():
            raise EditError("file_not_found", f"File does not exist: {path.name}")
        if not path.is_file():
            raise EditError("not_a_file", f"Path is not a file: {path.name}")
        if path.stat().st_size > MAX_TEXT_FILE_BYTES:
            raise EditError(
                "file_too_large",
                f"File exceeds {MAX_TEXT_FILE_BYTES} bytes: {path.name}",
                retryable=False,
            )
        return cls.from_bytes(path.read_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> TextDocument:
        if len(data) > MAX_TEXT_FILE_BYTES:
            raise EditError(
                "file_too_large",
                f"Text exceeds {MAX_TEXT_FILE_BYTES} bytes",
                retryable=False,
            )
        if b"\x00" in data:
            raise EditError(
                "unsupported_text_encoding",
                "Binary files are not supported",
                retryable=False,
            )
        encoding: Literal["utf-8", "utf-8-sig"] = (
            "utf-8-sig" if data.startswith(UTF8_BOM) else "utf-8"
        )
        payload = data[len(UTF8_BOM) :] if encoding == "utf-8-sig" else data
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise EditError(
                "unsupported_text_encoding",
                "Only UTF-8 text files are supported",
                retryable=False,
            ) from None
        without_crlf = content.replace("\r\n", "")
        if "\r" in without_crlf or ("\r\n" in content and "\n" in without_crlf):
            raise EditError(
                "unsupported_newline_style",
                "Mixed or legacy CR newlines are not supported",
                retryable=False,
            )
        newline_style: Literal["lf", "crlf"] = "crlf" if "\r\n" in content else "lf"
        normalized = content.replace("\r\n", "\n")
        return cls(
            raw_bytes=data,
            normalized_content=normalized,
            encoding=encoding,
            newline_style=newline_style,
            has_trailing_newline=normalized.endswith("\n"),
        )

    @property
    def lines(self) -> list[str]:
        return split_normalized_lines(self.normalized_content)

    @property
    def total_lines(self) -> int:
        return len(self.lines)

    @property
    def normalized_hash(self) -> str:
        return hashlib.sha256(self.normalized_content.encode("utf-8")).hexdigest()

    @property
    def raw_hash(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()

    def to_snapshot(
        self,
        *,
        relative_path: str,
        parent_snapshot_id: str | None = None,
    ) -> FileSnapshot:
        snapshot_id = hashlib.sha256(f"{relative_path}\0{self.raw_hash}".encode()).hexdigest()
        return FileSnapshot(
            snapshot_id=snapshot_id,
            display_tag=snapshot_id[:8].upper(),
            relative_path=relative_path,
            normalized_content_hash=self.normalized_hash,
            raw_bytes_hash=self.raw_hash,
            encoding=self.encoding,
            newline_style=self.newline_style,
            has_trailing_newline=self.has_trailing_newline,
            total_lines=self.total_lines,
            normalized_content=self.normalized_content,
            parent_snapshot_id=parent_snapshot_id,
        )

    @classmethod
    def from_normalized(
        cls,
        content: str,
        *,
        encoding: Literal["utf-8", "utf-8-sig"],
        newline_style: Literal["lf", "crlf"],
    ) -> TextDocument:
        serialized = content.replace("\n", "\r\n") if newline_style == "crlf" else content
        data = serialized.encode("utf-8")
        if encoding == "utf-8-sig":
            data = UTF8_BOM + data
        return cls.from_bytes(data)


def split_normalized_lines(content: str) -> list[str]:
    """Split only on normalized LF, leaving other Unicode separators untouched."""
    if not content:
        return []
    body = content[:-1] if content.endswith("\n") else content
    return body.split("\n")
