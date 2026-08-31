"""Small durable SQLite index for file, symbol, reference, import, and scope facts."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from repo_rivet.semantic.models import (
    ImportRecord,
    IndexedFile,
    ReferenceRecord,
    ScopeRecord,
    SemanticDiagnostic,
    SymbolRecord,
)
from repo_rivet.semantic.symbol_extractor import ExtractedDocument

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO meta(key, value) VALUES ('index_revision', 0);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    workspace_revision INTEGER NOT NULL,
    parse_error_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS symbols (
    symbol_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    qualified_name TEXT,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    scope_id TEXT,
    start_line INTEGER NOT NULL,
    start_column INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    end_column INTEGER NOT NULL,
    signature TEXT,
    documentation TEXT,
    exported INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS symbols_name_idx ON symbols(name);
CREATE INDEX IF NOT EXISTS symbols_path_idx ON symbols(path);
CREATE TABLE IF NOT EXISTS "references" (
    reference_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    scope_id TEXT,
    line INTEGER NOT NULL,
    column_number INTEGER NOT NULL,
    context_kind TEXT NOT NULL,
    snapshot_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS references_name_idx ON "references"(name);
CREATE INDEX IF NOT EXISTS references_path_idx ON "references"(path);
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    imported_name TEXT NOT NULL,
    local_alias TEXT,
    target_module TEXT NOT NULL,
    line INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS imports_path_idx ON imports(source_path);
CREATE TABLE IF NOT EXISTS scopes (
    scope_id TEXT PRIMARY KEY,
    parent_scope_id TEXT,
    kind TEXT NOT NULL,
    name TEXT,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS scopes_path_idx ON scopes(path);
CREATE TABLE IF NOT EXISTS diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    line INTEGER,
    column_number INTEGER,
    severity TEXT NOT NULL,
    source TEXT NOT NULL,
    code TEXT,
    message TEXT NOT NULL,
    workspace_revision INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS diagnostics_path_idx ON diagnostics(path);
"""


class SemanticIndexStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(path) if path is not None else ":memory:",
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.executescript(_SCHEMA)

    @property
    def index_revision(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM meta WHERE key = 'index_revision'"
            ).fetchone()
        return int(row["value"]) if row is not None else 0

    def indexed_file(self, path: str) -> IndexedFile | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        if row is None:
            return None
        return IndexedFile(
            path=row["path"],
            language=row["language"],
            snapshot_id=row["snapshot_id"],
            content_hash=row["content_hash"],
            indexed_at_workspace_revision=row["workspace_revision"],
            parse_error_count=row["parse_error_count"],
        )

    def replace_file(self, file: IndexedFile, extracted: ExtractedDocument) -> int:
        with self._lock, self._connection:
            self._delete_path(file.path)
            self._connection.execute(
                """
                INSERT INTO files(
                    path, language, snapshot_id, content_hash,
                    workspace_revision, parse_error_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file.path,
                    file.language,
                    file.snapshot_id,
                    file.content_hash,
                    file.indexed_at_workspace_revision,
                    file.parse_error_count,
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO symbols(
                    symbol_id, name, qualified_name, kind, path, scope_id,
                    start_line, start_column, end_line, end_column,
                    signature, documentation, exported, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.symbol_id,
                        item.name,
                        item.qualified_name,
                        item.kind,
                        item.path,
                        item.scope_id,
                        item.start_line,
                        item.start_column,
                        item.end_line,
                        item.end_column,
                        item.signature,
                        item.documentation,
                        int(item.exported),
                        item.snapshot_id,
                    )
                    for item in extracted.symbols
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO "references"(
                    reference_id, name, path, scope_id, line,
                    column_number, context_kind, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.reference_id,
                        item.name,
                        item.path,
                        item.scope_id,
                        item.line,
                        item.column,
                        item.context_kind,
                        item.snapshot_id,
                    )
                    for item in extracted.references
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO imports(
                    source_path, imported_name, local_alias,
                    target_module, line, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.source_path,
                        item.imported_name,
                        item.local_alias,
                        item.target_module,
                        item.line,
                        item.snapshot_id,
                    )
                    for item in extracted.imports
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO scopes(
                    scope_id, parent_scope_id, kind, name, path,
                    start_line, end_line, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.scope_id,
                        item.parent_scope_id,
                        item.kind,
                        item.name,
                        item.path,
                        item.start_line,
                        item.end_line,
                        item.snapshot_id,
                    )
                    for item in extracted.scopes
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO diagnostics(
                    path, line, column_number, severity, source,
                    code, message, workspace_revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.path,
                        item.line,
                        item.column,
                        item.severity,
                        item.source,
                        item.code,
                        item.message,
                        item.workspace_revision,
                    )
                    for item in extracted.diagnostics
                ],
            )
            return self._increment_revision()

    def remove_missing(self, existing_paths: set[str]) -> int:
        with self._lock:
            indexed = {
                str(row["path"])
                for row in self._connection.execute("SELECT path FROM files").fetchall()
            }
            missing = indexed - existing_paths
            if not missing:
                return self.index_revision
            with self._connection:
                for path in missing:
                    self._delete_path(path)
                return self._increment_revision()

    def symbols_for_path(self, path: str) -> list[SymbolRecord]:
        return self._symbols(
            "SELECT * FROM symbols WHERE path = ? ORDER BY start_line, start_column",
            (path,),
        )

    def search_symbols(self, query: str, *, limit: int = 100) -> list[SymbolRecord]:
        escaped = query.replace("%", "\\%").replace("_", "\\_")
        return self._symbols(
            """
            SELECT * FROM symbols
            WHERE name = ? OR name LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\'
            ORDER BY
                CASE WHEN name = ? THEN 0 WHEN name LIKE ? ESCAPE '\\' THEN 1 ELSE 2 END,
                length(name), path, start_line
            LIMIT ?
            """,
            (query, f"{escaped}%", f"%{escaped}%", query, f"{escaped}%", limit),
        )

    def references_for_name(self, name: str, *, limit: int = 200) -> list[ReferenceRecord]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM "references"
                WHERE name = ?
                ORDER BY path, line, column_number
                LIMIT ?
                """,
                (name, limit),
            ).fetchall()
        return [
            ReferenceRecord(
                reference_id=row["reference_id"],
                name=row["name"],
                path=row["path"],
                snapshot_id=row["snapshot_id"],
                line=row["line"],
                column=row["column_number"],
                scope_id=row["scope_id"],
                context_kind=row["context_kind"],
            )
            for row in rows
        ]

    def imports_for_path(self, path: str) -> list[ImportRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM imports WHERE source_path = ? ORDER BY line",
                (path,),
            ).fetchall()
        return [
            ImportRecord(
                source_path=row["source_path"],
                imported_name=row["imported_name"],
                local_alias=row["local_alias"],
                target_module=row["target_module"],
                line=row["line"],
                snapshot_id=row["snapshot_id"],
            )
            for row in rows
        ]

    def scope(self, scope_id: str | None) -> ScopeRecord | None:
        if scope_id is None:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM scopes WHERE scope_id = ?",
                (scope_id,),
            ).fetchone()
        if row is None:
            return None
        return ScopeRecord(
            scope_id=row["scope_id"],
            parent_scope_id=row["parent_scope_id"],
            kind=row["kind"],
            name=row["name"],
            path=row["path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            snapshot_id=row["snapshot_id"],
        )

    def diagnostics(self, path: str | None = None) -> list[SemanticDiagnostic]:
        sql = "SELECT * FROM diagnostics"
        values: tuple[Any, ...] = ()
        if path is not None:
            sql += " WHERE path = ?"
            values = (path,)
        sql += " ORDER BY path, line, column_number LIMIT 200"
        with self._lock:
            rows = self._connection.execute(sql, values).fetchall()
        return [
            SemanticDiagnostic(
                path=row["path"],
                line=row["line"],
                column=row["column_number"],
                severity=row["severity"],
                source=row["source"],
                code=row["code"],
                message=row["message"],
                workspace_revision=row["workspace_revision"],
            )
            for row in rows
        ]

    def _symbols(self, sql: str, values: tuple[Any, ...]) -> list[SymbolRecord]:
        with self._lock:
            rows = self._connection.execute(sql, values).fetchall()
        return [
            SymbolRecord(
                symbol_id=row["symbol_id"],
                name=row["name"],
                qualified_name=row["qualified_name"],
                kind=row["kind"],
                path=row["path"],
                scope_id=row["scope_id"],
                start_line=row["start_line"],
                start_column=row["start_column"],
                end_line=row["end_line"],
                end_column=row["end_column"],
                signature=row["signature"],
                documentation=row["documentation"],
                exported=bool(row["exported"]),
                snapshot_id=row["snapshot_id"],
            )
            for row in rows
        ]

    def _delete_path(self, path: str) -> None:
        self._connection.execute("DELETE FROM files WHERE path = ?", (path,))
        self._connection.execute("DELETE FROM symbols WHERE path = ?", (path,))
        self._connection.execute('DELETE FROM "references" WHERE path = ?', (path,))
        self._connection.execute("DELETE FROM imports WHERE source_path = ?", (path,))
        self._connection.execute("DELETE FROM scopes WHERE path = ?", (path,))
        self._connection.execute("DELETE FROM diagnostics WHERE path = ?", (path,))

    def _increment_revision(self) -> int:
        self._connection.execute("UPDATE meta SET value = value + 1 WHERE key = 'index_revision'")
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = 'index_revision'"
        ).fetchone()
        return int(row["value"])
