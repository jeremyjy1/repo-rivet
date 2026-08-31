"""Lazy semantic indexing and confidence-ranked query routing."""

from __future__ import annotations

import re
from pathlib import Path

from tree_sitter import Node, Point

from repo_rivet.editing.document import TextDocument
from repo_rivet.editing.runtime import EditingRuntime
from repo_rivet.safety.path_policy import WorkspacePathPolicy
from repo_rivet.semantic.index_store import SemanticIndexStore
from repo_rivet.semantic.models import (
    Confidence,
    IndexedFile,
    QueryPrecision,
    QueryStatus,
    ReferenceRecord,
    SemanticAction,
    SemanticDiagnostic,
    SemanticLocation,
    SemanticQueryArguments,
    SemanticQueryResult,
    SymbolRecord,
)
from repo_rivet.semantic.parser_registry import ParsedDocument, ParserRegistry
from repo_rivet.semantic.symbol_extractor import extract_document

_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".local",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".reporivet",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "venv",
        "web_dist",
    }
)
_SENSITIVE_NAMES = frozenset({".env", ".npmrc", ".pypirc", "reporivet.toml"})
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_MAX_INDEXED_FILES = 5_000
_MAX_TEXT_CANDIDATES = 50


class SemanticEngine:
    def __init__(
        self,
        path_policy: WorkspacePathPolicy,
        editing_runtime: EditingRuntime,
        *,
        index_path: Path | None = None,
        scan_roots: list[str] | None = None,
    ) -> None:
        self.path_policy = path_policy
        self.editing_runtime = editing_runtime
        self.parsers = ParserRegistry()
        self.index = SemanticIndexStore(index_path)
        self.scan_roots = tuple(scan_roots or ["."])

    def query(self, arguments: SemanticQueryArguments) -> SemanticQueryResult:
        action = arguments.action
        if action == SemanticAction.SYMBOLS:
            return self._file_symbols(arguments)
        if action == SemanticAction.WORKSPACE_SYMBOLS:
            return self._workspace_symbols(arguments)
        if action == SemanticAction.DEFINITION:
            return self._definitions(arguments)
        if action == SemanticAction.REFERENCES:
            return self._references(arguments)
        return self._diagnostics(arguments)

    def _file_symbols(self, arguments: SemanticQueryArguments) -> SemanticQueryResult:
        path = str(arguments.path)
        indexed = self._ensure_file(path)
        if indexed is None:
            return self._unsupported(arguments, path)
        symbols = self.index.symbols_for_path(indexed.path)
        locations = [
            self._symbol_location(item, Confidence.EXACT, "declared in requested file")
            for item in symbols
        ]
        return self._result(
            arguments,
            status=QueryStatus.EXACT,
            confidence=Confidence.EXACT,
            results=locations,
            snapshots={indexed.path: indexed.snapshot_id},
        )

    def _workspace_symbols(self, arguments: SemanticQueryArguments) -> SemanticQueryResult:
        self._refresh_workspace()
        query = str(arguments.query or arguments.symbol)
        symbols = self.index.search_symbols(query)
        results: list[SemanticLocation] = []
        for symbol in symbols:
            if symbol.name == query:
                confidence = Confidence.EXACT
                reason = "exact symbol-name match"
            elif symbol.name.casefold().startswith(query.casefold()):
                confidence = Confidence.HIGH
                reason = "symbol-name prefix match"
            else:
                confidence = Confidence.MEDIUM
                reason = "symbol-name substring match"
            results.append(self._symbol_location(symbol, confidence, reason))
        exact_count = sum(item.name == query for item in symbols)
        status = (
            QueryStatus.EXACT
            if exact_count == 1 and len(symbols) == 1
            else QueryStatus.AMBIGUOUS
            if symbols
            else QueryStatus.PARTIAL
        )
        confidence = results[0].confidence if results else Confidence.LOW
        return self._result(
            arguments,
            status=status,
            confidence=confidence,
            results=results,
            snapshots=self._snapshots(results),
            warnings=[] if results else ["No indexed symbol matched the query."],
        )

    def _definitions(self, arguments: SemanticQueryArguments) -> SemanticQueryResult:
        source_path, name, source_scope = self._resolve_target(arguments)
        if arguments.precision == QueryPrecision.TEXT:
            text_results = self._text_candidates(name, seen=set())
            return self._result(
                arguments,
                status=QueryStatus.PARTIAL,
                confidence=Confidence.LOW,
                results=text_results,
                snapshots=self._snapshots(text_results),
                warnings=["Text matches are not proven symbol definitions."],
            )
        self._refresh_workspace()
        symbols = [item for item in self.index.search_symbols(name) if item.name == name]
        imports = self.index.imports_for_path(source_path) if source_path else []
        imported_names = {
            item.imported_name
            for item in imports
            if (item.local_alias or item.imported_name) == name
        }
        if imported_names:
            symbols.extend(
                item
                for imported_name in imported_names
                for item in self.index.search_symbols(imported_name)
                if item.name == imported_name and item not in symbols
            )
        ranked = sorted(
            (
                self._rank_definition(
                    symbol,
                    source_path=source_path,
                    source_scope=source_scope,
                    imported=symbol.name in imported_names,
                )
                for symbol in symbols
            ),
            key=lambda item: (_confidence_rank(item.confidence), item.path, item.start_line),
        )[:30]
        if len(ranked) == 1 and ranked[0].confidence in {Confidence.EXACT, Confidence.HIGH}:
            status = QueryStatus.EXACT
        elif ranked:
            status = QueryStatus.AMBIGUOUS
        else:
            status = QueryStatus.PARTIAL
        textual: list[SemanticLocation] = []
        if not ranked and arguments.precision != QueryPrecision.SYNTAX:
            textual = self._text_candidates(name, seen=set())
        results = [*ranked, *textual]
        warnings = [] if ranked else [f"No syntax-indexed definition found for {name!r}."]
        if textual:
            warnings.append("Returned text matches are not proven symbol definitions.")
        if arguments.precision == QueryPrecision.COMPILER:
            warnings.append(
                "Compiler-level definition resolution is unavailable; use these candidates "
                "as navigation evidence."
            )
        return self._result(
            arguments,
            status=status,
            confidence=results[0].confidence if results else Confidence.LOW,
            results=results,
            snapshots=self._snapshots(results),
            warnings=warnings,
        )

    def _references(self, arguments: SemanticQueryArguments) -> SemanticQueryResult:
        source_path, name, source_scope = self._resolve_target(arguments)
        if arguments.precision == QueryPrecision.TEXT:
            textual = self._text_candidates(name, seen=set())
            return self._result(
                arguments,
                status=QueryStatus.PARTIAL,
                confidence=Confidence.LOW,
                results=textual,
                snapshots=self._snapshots(textual),
                warnings=["Text matches may be comments, strings, or unrelated names."],
            )
        self._refresh_workspace()
        definitions = [item for item in self.index.search_symbols(name) if item.name == name]
        target = self._best_target(definitions, source_path=source_path, source_scope=source_scope)
        syntax_results = [
            self._rank_reference(item, target=target)
            for item in self.index.references_for_name(name)
        ]
        syntax_results.sort(
            key=lambda item: (_confidence_rank(item.confidence), item.path, item.start_line)
        )
        seen = {(item.path, item.start_line, item.start_column) for item in syntax_results}
        textual = (
            self._text_candidates(name, seen=seen)
            if not syntax_results and arguments.precision != QueryPrecision.SYNTAX
            else []
        )
        results = [*syntax_results, *textual][:200]
        confidence = results[0].confidence if results else Confidence.LOW
        warnings: list[str] = []
        if len(definitions) != 1:
            warnings.append(
                f"The target symbol has {len(definitions)} definition candidates; "
                "global references are not claimed as exact."
            )
        if textual:
            warnings.append(
                "Text candidates may be comments, strings, generated code, or unrelated names."
            )
        if arguments.precision == QueryPrecision.COMPILER:
            warnings.append(
                "Compiler-level reference resolution is unavailable; ambiguous candidates "
                "remain explicitly ranked."
            )
        uncertain = any(item.confidence in {Confidence.MEDIUM, Confidence.LOW} for item in results)
        return self._result(
            arguments,
            status=(
                QueryStatus.PARTIAL
                if textual or len(definitions) != 1 or uncertain
                else QueryStatus.EXACT
            ),
            confidence=confidence,
            results=results,
            snapshots=self._snapshots(results),
            warnings=warnings,
        )

    def _diagnostics(self, arguments: SemanticQueryArguments) -> SemanticQueryResult:
        snapshots: dict[str, str] = {}
        if arguments.path:
            indexed = self._ensure_file(arguments.path)
            if indexed is None:
                return self._unsupported(arguments, arguments.path)
            snapshots[indexed.path] = indexed.snapshot_id
            diagnostics = self.index.diagnostics(indexed.path)
        else:
            indexed_files = self._refresh_workspace()
            snapshots = {item.path: item.snapshot_id for item in indexed_files}
            diagnostics = self.index.diagnostics()
        diagnostics = [
            item.model_copy(update={"workspace_revision": self.workspace_revision})
            for item in diagnostics
        ]
        results = [
            self._diagnostic_location(item, snapshots.get(item.path)) for item in diagnostics
        ]
        # Only return snapshots referenced by actual diagnostics. A clean workspace must not
        # inject thousands of unrelated file IDs into the model context.
        snapshots = self._snapshots(results)
        warnings: list[str] = []
        status = QueryStatus.EXACT
        confidence = Confidence.EXACT
        if arguments.precision == QueryPrecision.COMPILER:
            status = QueryStatus.PARTIAL
            confidence = Confidence.LOW
            warnings.append(
                "Compiler diagnostics require a registered verification check; semantic_query "
                "does not execute project code or bypass approval."
            )
        warnings.append("Syntax diagnostics do not prove type correctness or verification success.")
        return self._result(
            arguments,
            status=status,
            confidence=confidence,
            results=results,
            snapshots=snapshots,
            warnings=warnings,
        )

    @property
    def workspace_revision(self) -> int:
        return self.editing_runtime.workspace_revision

    def _ensure_file(self, user_path: str) -> IndexedFile | None:
        resolved = self.path_policy.resolve(user_path)
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"Semantic path is not a file: {user_path}")
        relative = resolved.relative_to(self.path_policy.workspace).as_posix()
        if resolved.name in _SENSITIVE_NAMES or resolved.name.startswith(".env"):
            raise ValueError(f"Sensitive configuration files cannot be indexed: {relative}")
        language = self.parsers.language_for_path(relative)
        if language is None:
            return None
        document = TextDocument.load(resolved)
        existing = self.index.indexed_file(relative)
        if existing is not None and existing.content_hash == document.raw_hash:
            return existing
        snapshot = self.editing_runtime.capture_document(
            relative,
            document,
            start_line=1,
            end_line=max(1, document.total_lines),
            source="semantic_query",
            visible=False,
        )
        parsed = self.parsers.parse(relative, document.raw_bytes)
        extracted = extract_document(
            parsed,
            snapshot_id=snapshot.snapshot_id,
            workspace_revision=self.workspace_revision,
        )
        indexed = IndexedFile(
            path=relative,
            language=language,
            snapshot_id=snapshot.snapshot_id,
            content_hash=document.raw_hash,
            indexed_at_workspace_revision=self.workspace_revision,
            parse_error_count=len(extracted.diagnostics),
        )
        self.index.replace_file(indexed, extracted)
        return indexed

    def _refresh_workspace(self) -> list[IndexedFile]:
        indexed: list[IndexedFile] = []
        existing_paths: set[str] = set()
        for path in self._semantic_files():
            relative = path.relative_to(self.path_policy.workspace).as_posix()
            existing_paths.add(relative)
            try:
                value = self._ensure_file(relative)
            except (OSError, UnicodeError, ValueError):
                continue
            if value is not None:
                indexed.append(value)
        self.index.remove_missing(existing_paths)
        return indexed

    def _semantic_files(self) -> list[Path]:
        values: list[Path] = []
        for path in self._candidate_paths():
            if len(values) >= _MAX_INDEXED_FILES:
                break
            try:
                relative = path.relative_to(self.path_policy.workspace)
            except ValueError:
                continue
            if any(part in _EXCLUDED_DIRECTORIES for part in relative.parts):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            if self.parsers.language_for_path(relative.as_posix()) is not None:
                values.append(path)
        return values

    def _resolve_target(
        self,
        arguments: SemanticQueryArguments,
    ) -> tuple[str | None, str, str | None]:
        if arguments.symbol:
            source_path: str | None = None
            if arguments.path is not None:
                resolved = self.path_policy.resolve(arguments.path)
                if not resolved.exists() or not resolved.is_file():
                    raise ValueError(f"Semantic path is not a file: {arguments.path}")
                source_path = resolved.relative_to(self.path_policy.workspace).as_posix()
                self._ensure_file(source_path)
            return source_path, arguments.symbol, None
        path = str(arguments.path)
        indexed = self._ensure_file(path)
        if indexed is None:
            raise ValueError(f"Unsupported semantic language: {path}")
        resolved = self.path_policy.resolve(path)
        document = TextDocument.load(resolved)
        parsed = self.parsers.parse(indexed.path, document.raw_bytes)
        if arguments.line is None or arguments.column is None:
            raise ValueError("A source position requires both line and column")
        node = _identifier_at(parsed, line=arguments.line, column=arguments.column)
        if node is None:
            raise ValueError(
                f"No identifier exists at {indexed.path}:{arguments.line}:{arguments.column}"
            )
        name = parsed.source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        references = [
            item
            for item in self.index.references_for_name(name)
            if item.path == indexed.path
            and item.line == node.start_point.row + 1
            and item.column == node.start_point.column + 1
        ]
        source_scope = references[0].scope_id if references else None
        return indexed.path, name, source_scope

    def _rank_definition(
        self,
        symbol: SymbolRecord,
        *,
        source_path: str | None,
        source_scope: str | None,
        imported: bool,
    ) -> SemanticLocation:
        if source_scope is not None and symbol.scope_id == source_scope:
            confidence = Confidence.EXACT
            reason = "same lexical scope and exact name"
        elif imported:
            confidence = Confidence.HIGH
            reason = "explicit import alias resolves to this symbol name"
        elif source_path is not None and symbol.path == source_path:
            confidence = Confidence.HIGH
            reason = "same file and exact name"
        elif symbol.exported:
            confidence = Confidence.MEDIUM
            reason = "exported workspace symbol with exact name"
        else:
            confidence = Confidence.LOW
            reason = "global workspace symbol with exact name only"
        return self._symbol_location(symbol, confidence, reason)

    def _rank_reference(
        self,
        reference: ReferenceRecord,
        *,
        target: SymbolRecord | None,
    ) -> SemanticLocation:
        if (
            target is not None
            and reference.path == target.path
            and reference.scope_id == target.scope_id
        ):
            confidence = Confidence.EXACT
            reason = "same resolved lexical scope"
        elif target is not None and reference.path == target.path:
            confidence = Confidence.HIGH
            reason = "same file as the unique definition"
        elif target is not None:
            confidence = Confidence.MEDIUM
            reason = "syntax-filtered workspace name candidate"
        else:
            confidence = Confidence.LOW
            reason = "syntax-filtered reference to an ambiguous symbol name"
        return SemanticLocation(
            path=reference.path,
            snapshot_id=reference.snapshot_id,
            name=reference.name,
            kind="reference",
            start_line=reference.line,
            start_column=reference.column,
            confidence=confidence,
            source="tree_sitter",
            reason=reason,
            context_kind=reference.context_kind,
        )

    def _text_candidates(
        self,
        name: str,
        *,
        seen: set[tuple[str, int, int]],
    ) -> list[SemanticLocation]:
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        results: list[SemanticLocation] = []
        for path in self._text_files():
            relative = path.relative_to(self.path_policy.workspace).as_posix()
            indexed = self.index.indexed_file(relative)
            try:
                document = TextDocument.load(path)
            except (OSError, UnicodeError, ValueError):
                continue
            if indexed is not None:
                snapshot_id = indexed.snapshot_id
            else:
                snapshot_id = self.editing_runtime.capture_document(
                    relative,
                    document,
                    start_line=1,
                    end_line=max(1, document.total_lines),
                    source="semantic_query",
                    visible=False,
                ).snapshot_id
            for line_number, line in enumerate(document.lines, start=1):
                for match in pattern.finditer(line):
                    location = (relative, line_number, match.start() + 1)
                    if location in seen:
                        continue
                    results.append(
                        SemanticLocation(
                            path=relative,
                            snapshot_id=snapshot_id,
                            name=name,
                            kind="text_candidate",
                            start_line=line_number,
                            start_column=match.start() + 1,
                            confidence=Confidence.LOW,
                            source="text_search",
                            reason="text match not proven by syntax or scope",
                        )
                    )
                    if len(results) >= _MAX_TEXT_CANDIDATES:
                        return results
        return results

    def _text_files(self) -> list[Path]:
        values: list[Path] = []
        for path in self._candidate_paths():
            try:
                relative = path.relative_to(self.path_policy.workspace)
            except ValueError:
                continue
            if any(part in _EXCLUDED_DIRECTORIES for part in relative.parts):
                continue
            if path.name in _SENSITIVE_NAMES or path.name.startswith(".env"):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            values.append(path)
            if len(values) >= _MAX_INDEXED_FILES:
                break
        return values

    def _candidate_paths(self) -> list[Path]:
        candidates: set[Path] = set()
        for root in self.scan_roots:
            resolved = self.path_policy.resolve(root)
            if resolved.is_file():
                candidates.add(resolved)
            elif resolved.is_dir():
                candidates.update(resolved.rglob("*"))
        return sorted(candidates)

    @staticmethod
    def _best_target(
        symbols: list[SymbolRecord],
        *,
        source_path: str | None,
        source_scope: str | None,
    ) -> SymbolRecord | None:
        if len(symbols) == 1:
            return symbols[0]
        scoped = [item for item in symbols if source_scope and item.scope_id == source_scope]
        if len(scoped) == 1:
            return scoped[0]
        local = [item for item in symbols if source_path and item.path == source_path]
        return local[0] if len(local) == 1 else None

    @staticmethod
    def _symbol_location(
        symbol: SymbolRecord,
        confidence: Confidence,
        reason: str,
    ) -> SemanticLocation:
        return SemanticLocation(
            path=symbol.path,
            snapshot_id=symbol.snapshot_id,
            name=symbol.name,
            kind=symbol.kind,
            start_line=symbol.start_line,
            start_column=symbol.start_column,
            end_line=symbol.end_line,
            end_column=symbol.end_column,
            confidence=confidence,
            source="tree_sitter",
            reason=reason,
            signature=symbol.signature,
            documentation=symbol.documentation,
        )

    @staticmethod
    def _diagnostic_location(
        diagnostic: SemanticDiagnostic,
        snapshot_id: str | None,
    ) -> SemanticLocation:
        return SemanticLocation(
            path=diagnostic.path,
            snapshot_id=snapshot_id,
            kind="diagnostic",
            start_line=diagnostic.line or 1,
            start_column=diagnostic.column or 1,
            confidence=Confidence.EXACT,
            source="tree_sitter",
            reason=diagnostic.message,
            diagnostic=diagnostic,
        )

    def _unsupported(self, arguments: SemanticQueryArguments, path: str) -> SemanticQueryResult:
        return self._result(
            arguments,
            status=QueryStatus.UNSUPPORTED,
            confidence=Confidence.LOW,
            results=[],
            snapshots={},
            warnings=[f"No Tree-sitter adapter is configured for {path}. Use search_text."],
        )

    def _result(
        self,
        arguments: SemanticQueryArguments,
        *,
        status: QueryStatus,
        confidence: Confidence,
        results: list[SemanticLocation],
        snapshots: dict[str, str],
        warnings: list[str] | None = None,
    ) -> SemanticQueryResult:
        return SemanticQueryResult(
            action=arguments.action,
            status=status,
            results=results,
            confidence=confidence,
            warnings=warnings or [],
            workspace_revision=self.workspace_revision,
            index_revision=self.index.index_revision,
            snapshot_ids=snapshots,
        )

    @staticmethod
    def _snapshots(results: list[SemanticLocation]) -> dict[str, str]:
        return {item.path: item.snapshot_id for item in results if item.snapshot_id is not None}


def _identifier_at(document: ParsedDocument, *, line: int, column: int) -> Node | None:
    source_lines = document.source.splitlines()
    source_line = source_lines[line - 1] if line <= len(source_lines) else b""
    line_text = source_line.decode("utf-8", errors="replace")
    byte_column = len(line_text[: max(0, column - 1)].encode("utf-8"))
    point = Point(max(0, line - 1), byte_column)
    node = document.tree.root_node.descendant_for_point_range(point, point)
    current: Node | None = node
    while current is not None:
        if current.type in {
            "identifier",
            "field_identifier",
            "property_identifier",
            "private_property_identifier",
            "type_identifier",
            "namespace_identifier",
        }:
            return current
        current = current.parent
    for match in _IDENTIFIER.finditer(line_text):
        if match.start() <= column - 1 <= match.end():
            start = Point(line - 1, len(line_text[: match.start()].encode("utf-8")))
            end = Point(line - 1, len(line_text[: match.end()].encode("utf-8")))
            fallback = document.tree.root_node.descendant_for_point_range(start, end)
            if fallback is not None and fallback.type in {
                "identifier",
                "type_identifier",
                "property_identifier",
            }:
                return fallback
    return None


def _confidence_rank(confidence: Confidence) -> int:
    return {
        Confidence.EXACT: 0,
        Confidence.HIGH: 1,
        Confidence.MEDIUM: 2,
        Confidence.LOW: 3,
    }[confidence]
