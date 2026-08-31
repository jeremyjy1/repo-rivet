"""Language-aware extraction from Tree-sitter concrete syntax trees."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from tree_sitter import Node

from repo_rivet.semantic.models import (
    ImportRecord,
    ReferenceKind,
    ReferenceRecord,
    ScopeRecord,
    SemanticDiagnostic,
    SymbolKind,
    SymbolRecord,
)
from repo_rivet.semantic.parser_registry import ParsedDocument

_IDENTIFIER_TYPES = frozenset(
    {
        "identifier",
        "field_identifier",
        "property_identifier",
        "private_property_identifier",
        "type_identifier",
        "namespace_identifier",
        "shorthand_property_identifier",
        "shorthand_property_identifier_pattern",
    }
)
_IMPORT_TYPES = frozenset(
    {"import_statement", "import_from_statement", "preproc_include", "require_call"}
)


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    symbols: list[SymbolRecord]
    references: list[ReferenceRecord]
    imports: list[ImportRecord]
    scopes: list[ScopeRecord]
    diagnostics: list[SemanticDiagnostic]


def extract_document(
    document: ParsedDocument,
    *,
    snapshot_id: str,
    workspace_revision: int,
) -> ExtractedDocument:
    extractor = _Extractor(document, snapshot_id=snapshot_id)
    extractor.walk(document.tree.root_node, parent_scope=None, scope_names=())
    diagnostics = _syntax_diagnostics(document, workspace_revision=workspace_revision)
    return ExtractedDocument(
        symbols=extractor.symbols,
        references=extractor.references,
        imports=extractor.imports,
        scopes=extractor.scopes,
        diagnostics=diagnostics,
    )


class _Extractor:
    def __init__(self, document: ParsedDocument, *, snapshot_id: str) -> None:
        self.document = document
        self.snapshot_id = snapshot_id
        self.symbols: list[SymbolRecord] = []
        self.references: list[ReferenceRecord] = []
        self.imports: list[ImportRecord] = []
        self.scopes: list[ScopeRecord] = []
        self._definition_spans: set[tuple[int, int]] = set()

    def walk(
        self,
        node: Node,
        *,
        parent_scope: str | None,
        scope_names: tuple[str, ...],
    ) -> None:
        definition = self._definition(node, parent_scope=parent_scope)
        node_scope = parent_scope
        child_scope_names = scope_names
        if definition is not None:
            name_node, kind = definition
            name = self._text(name_node)
            if name:
                self._definition_spans.add((name_node.start_byte, name_node.end_byte))
                exported = self._exported(node, parent_scope=parent_scope)
                symbol = self._symbol(
                    node,
                    name_node=name_node,
                    name=name,
                    kind=kind,
                    parent_scope=parent_scope,
                    scope_names=scope_names,
                    exported=exported,
                )
                self.symbols.append(symbol)
                if self._is_scope_node(node):
                    node_scope = self._scope(node, name=name, parent_scope=parent_scope)
                    child_scope_names = (*scope_names, name)

        if node.type in _IMPORT_TYPES:
            self.imports.extend(self._imports(node))

        for child in node.named_children:
            self.walk(
                child,
                parent_scope=node_scope,
                scope_names=child_scope_names,
            )

        if (
            node.type in _IDENTIFIER_TYPES
            and (
                node.start_byte,
                node.end_byte,
            )
            not in self._definition_spans
        ):
            name = self._text(node)
            if name:
                self.references.append(self._reference(node, name=name, scope_id=parent_scope))

    def _definition(
        self,
        node: Node,
        *,
        parent_scope: str | None,
    ) -> tuple[Node, SymbolKind] | None:
        language = self.document.language
        if language == "python":
            if node.type == "class_definition":
                name = _field_optional(node, "name")
                return (name, "class") if name is not None else None
            if node.type == "function_definition":
                kind: SymbolKind = "method" if parent_scope is not None else "function"
                name = _field_optional(node, "name")
                return (name, kind) if name is not None else None
            if node.type == "assignment":
                name = _first_identifier(_field_optional(node, "left"))
                if name is not None:
                    return name, "constant" if self._text(name).isupper() else "variable"
            return None

        if language in {"c", "cpp"}:
            if node.type == "namespace_definition":
                name = _field_optional(node, "name") or _first_child(node, "namespace_identifier")
                return (name, "namespace") if name is not None else None
            if node.type == "class_specifier":
                name = _field_optional(node, "name")
                return (name, "class") if name is not None else None
            if node.type == "struct_specifier":
                name = _field_optional(node, "name")
                return (name, "struct") if name is not None else None
            if node.type == "function_definition":
                name = _declarator_name(_field_optional(node, "declarator"))
                if name is not None:
                    return name, "method" if parent_scope is not None else "function"
            if node.type in {"declaration", "field_declaration"}:
                declarator = _field_optional(node, "declarator")
                if declarator is None:
                    declarator = _first_declarator(node)
                if declarator is None:
                    return None
                function = _descendant_of_type(declarator, {"function_declarator"})
                if function is not None:
                    name = _declarator_name(function)
                    if name is not None:
                        return name, "method" if node.type == "field_declaration" else "function"
                name = _declarator_name(declarator)
                if name is not None:
                    kind = "field" if node.type == "field_declaration" else "variable"
                    return name, kind
            return None

        if node.type == "class_declaration":
            name = _field_optional(node, "name")
            return (name, "class") if name is not None else None
        if node.type == "interface_declaration":
            name = _field_optional(node, "name")
            return (name, "interface") if name is not None else None
        if node.type in {"function_declaration", "generator_function_declaration"}:
            name = _field_optional(node, "name")
            return (name, "function") if name is not None else None
        if node.type in {"method_definition", "method_signature"}:
            name = _field_optional(node, "name")
            return (name, "method") if name is not None else None
        if node.type in {"type_alias_declaration", "enum_declaration"}:
            name = _field_optional(node, "name")
            return (name, "interface") if name is not None else None
        if node.type == "variable_declarator":
            name = _first_identifier(_field_optional(node, "name"))
            if name is not None:
                return name, "constant" if self._text(name).isupper() else "variable"
        return None

    def _symbol(
        self,
        node: Node,
        *,
        name_node: Node,
        name: str,
        kind: SymbolKind,
        parent_scope: str | None,
        scope_names: tuple[str, ...],
        exported: bool,
    ) -> SymbolRecord:
        qualified = ".".join((*scope_names, name))
        symbol_id = _stable_id(
            "symbol",
            self.document.path,
            self.snapshot_id,
            str(name_node.start_byte),
            name,
            kind,
        )
        return SymbolRecord(
            symbol_id=symbol_id,
            name=name,
            kind=kind,
            path=self.document.path,
            snapshot_id=self.snapshot_id,
            start_line=name_node.start_point.row + 1,
            start_column=self._column(name_node, end=False),
            end_line=node.end_point.row + 1,
            end_column=self._column(node, end=True),
            scope_id=parent_scope,
            qualified_name=qualified,
            signature=self._signature(node),
            documentation=self._documentation(node),
            exported=exported,
        )

    def _scope(self, node: Node, *, name: str, parent_scope: str | None) -> str:
        scope_id = _stable_id(
            "scope",
            self.document.path,
            self.snapshot_id,
            str(node.start_byte),
            node.type,
            name,
        )
        self.scopes.append(
            ScopeRecord(
                scope_id=scope_id,
                parent_scope_id=parent_scope,
                kind=node.type,
                name=name,
                path=self.document.path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                snapshot_id=self.snapshot_id,
            )
        )
        return scope_id

    def _reference(self, node: Node, *, name: str, scope_id: str | None) -> ReferenceRecord:
        return ReferenceRecord(
            reference_id=_stable_id(
                "reference",
                self.document.path,
                self.snapshot_id,
                str(node.start_byte),
                name,
            ),
            name=name,
            path=self.document.path,
            snapshot_id=self.snapshot_id,
            line=node.start_point.row + 1,
            column=self._column(node, end=False),
            scope_id=scope_id,
            context_kind=self._reference_kind(node),
        )

    def _reference_kind(self, node: Node) -> ReferenceKind:
        ancestors = list(_ancestors(node, limit=4))
        types = {item.type for item in ancestors}
        parent = node.parent
        if types & _IMPORT_TYPES:
            return "import"
        if types & {"class_heritage", "extends_clause"}:
            return "inheritance"
        if node.type == "type_identifier" or types & {
            "type_annotation",
            "base_class_clause",
            "type_descriptor",
        }:
            return "type"
        if parent is not None and parent.type in {"call", "call_expression"}:
            function = _field_optional(parent, "function")
            if function is None or _contains(function, node):
                return "call"
        if types & {"assignment", "augmented_assignment", "variable_declarator"}:
            assignment = next(
                (
                    item
                    for item in ancestors
                    if item.type in {"assignment", "augmented_assignment", "variable_declarator"}
                ),
                None,
            )
            if assignment is not None:
                left = _field_optional(assignment, "left") or _field_optional(assignment, "name")
                if left is not None and _contains(left, node):
                    return "write"
        return "read"

    def _imports(self, node: Node) -> list[ImportRecord]:
        text = self._text(node)
        line = node.start_point.row + 1
        if self.document.language == "python":
            return _python_imports(
                text,
                source_path=self.document.path,
                line=line,
                snapshot_id=self.snapshot_id,
            )
        if self.document.language in {"c", "cpp"}:
            target = text.removeprefix("#include").strip().strip('<>"')
            return [
                ImportRecord(
                    source_path=self.document.path,
                    imported_name=target.rsplit("/", 1)[-1],
                    target_module=target,
                    line=line,
                    snapshot_id=self.snapshot_id,
                )
            ]
        source = _field_optional(node, "source") or _last_child_of_type(node, {"string"})
        target = self._text(source).strip("'\"") if source is not None else ""
        records: list[ImportRecord] = []
        for specifier in (child for child in _walk_nodes(node) if child.type == "import_specifier"):
            imported = _field_optional(specifier, "name")
            alias = _field_optional(specifier, "alias")
            if imported is None:
                continue
            records.append(
                ImportRecord(
                    source_path=self.document.path,
                    imported_name=self._text(imported),
                    local_alias=self._text(alias) if alias is not None else None,
                    target_module=target,
                    line=specifier.start_point.row + 1,
                    snapshot_id=self.snapshot_id,
                )
            )
        clause = _first_child(node, "import_clause")
        if clause is not None:
            for child in clause.named_children:
                if child.type == "identifier":
                    records.append(
                        ImportRecord(
                            source_path=self.document.path,
                            imported_name="default",
                            local_alias=self._text(child),
                            target_module=target,
                            line=child.start_point.row + 1,
                            snapshot_id=self.snapshot_id,
                        )
                    )
                elif child.type == "namespace_import":
                    alias = _first_identifier(child)
                    if alias is not None:
                        records.append(
                            ImportRecord(
                                source_path=self.document.path,
                                imported_name="*",
                                local_alias=self._text(alias),
                                target_module=target,
                                line=alias.start_point.row + 1,
                                snapshot_id=self.snapshot_id,
                            )
                        )
        if records:
            return records
        return [
            ImportRecord(
                source_path=self.document.path,
                imported_name=target,
                target_module=target,
                line=line,
                snapshot_id=self.snapshot_id,
            )
        ]

    def _signature(self, node: Node) -> str | None:
        text = self._text(node)
        if not text:
            return None
        first_line = text.splitlines()[0]
        for delimiter in ("{", ":"):
            if delimiter in first_line:
                first_line = first_line.split(delimiter, 1)[0] + delimiter
                break
        return " ".join(first_line.split())[:300] or None

    def _documentation(self, node: Node) -> str | None:
        previous = node.prev_named_sibling
        if previous is not None and previous.type in {"comment", "line_comment", "block_comment"}:
            return self._text(previous).strip()[:1_000] or None
        if self.document.language == "python":
            body = _field_optional(node, "body")
            if body is not None and body.named_children:
                first = body.named_children[0]
                if first.type == "expression_statement" and first.named_children:
                    string = first.named_children[0]
                    if string.type == "string":
                        return self._text(string).strip("'\"")[:1_000] or None
        return None

    def _exported(self, node: Node, *, parent_scope: str | None) -> bool:
        if self.document.language == "python":
            definition = self._definition(node, parent_scope=None)
            return bool(
                parent_scope is None
                and definition
                and not self._text(definition[0]).startswith("_")
            )
        return node.parent is not None and node.parent.type == "export_statement"

    def _is_scope_node(self, node: Node) -> bool:
        if self.document.language == "python":
            return node.type in {"class_definition", "function_definition"}
        if self.document.language in {"c", "cpp"}:
            return node.type in {
                "namespace_definition",
                "class_specifier",
                "struct_specifier",
                "function_definition",
            }
        return node.type in {
            "class_declaration",
            "interface_declaration",
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
        }

    def _text(self, node: Node) -> str:
        return self.document.source[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def _column(self, node: Node, *, end: bool) -> int:
        point = node.end_point if end else node.start_point
        lines = self.document.source.splitlines()
        line = lines[point.row] if point.row < len(lines) else b""
        prefix = line[: point.column]
        return len(prefix.decode("utf-8", errors="replace")) + 1


def _syntax_diagnostics(
    document: ParsedDocument,
    *,
    workspace_revision: int,
) -> list[SemanticDiagnostic]:
    diagnostics: list[SemanticDiagnostic] = []
    for node in _walk_all_nodes(document.tree.root_node):
        if not node.is_error and not node.is_missing:
            continue
        diagnostics.append(
            SemanticDiagnostic(
                path=document.path,
                line=node.start_point.row + 1,
                column=_display_column(
                    document.source, node.start_point.row, node.start_point.column
                ),
                severity="error",
                source="tree_sitter",
                code="missing_node" if node.is_missing else "parse_error",
                message=(
                    f"Missing syntax node: {node.type}"
                    if node.is_missing
                    else f"Unexpected syntax near {node.type}"
                ),
                workspace_revision=workspace_revision,
            )
        )
        if len(diagnostics) >= 100:
            break
    return diagnostics


def _python_imports(
    text: str,
    *,
    source_path: str,
    line: int,
    snapshot_id: str,
) -> list[ImportRecord]:
    records: list[ImportRecord] = []
    from_match = re.match(r"from\s+([.\w]+)\s+import\s+(.+)", text)
    if from_match:
        module, imported = from_match.groups()
        for item in imported.strip("() ").split(","):
            parts = item.strip().split(" as ", 1)
            if not parts[0]:
                continue
            records.append(
                ImportRecord(
                    source_path=source_path,
                    imported_name=parts[0].strip(),
                    local_alias=parts[-1].strip() if len(parts) == 2 else None,
                    target_module=module,
                    line=line,
                    snapshot_id=snapshot_id,
                )
            )
        return records
    imported = text.removeprefix("import ")
    for item in imported.split(","):
        parts = item.strip().split(" as ", 1)
        if not parts[0]:
            continue
        records.append(
            ImportRecord(
                source_path=source_path,
                imported_name=parts[0].strip(),
                local_alias=parts[-1].strip() if len(parts) == 2 else None,
                target_module=parts[0].strip(),
                line=line,
                snapshot_id=snapshot_id,
            )
        )
    return records


def _display_column(source: bytes, row: int, byte_column: int) -> int:
    lines = source.splitlines()
    line = lines[row] if row < len(lines) else b""
    return len(line[:byte_column].decode("utf-8", errors="replace")) + 1


def _field_optional(node: Node, name: str) -> Node | None:
    return node.child_by_field_name(name)


def _first_child(node: Node, node_type: str) -> Node | None:
    return next((child for child in node.named_children if child.type == node_type), None)


def _first_identifier(node: Node | None) -> Node | None:
    if node is None:
        return None
    if node.type in _IDENTIFIER_TYPES:
        return node
    for child in node.named_children:
        identifier = _first_identifier(child)
        if identifier is not None:
            return identifier
    return None


def _first_declarator(node: Node) -> Node | None:
    for child in node.named_children:
        if child.type in {
            "identifier",
            "field_identifier",
            "function_declarator",
            "pointer_declarator",
            "reference_declarator",
            "init_declarator",
            "array_declarator",
        }:
            return child
    return None


def _declarator_name(node: Node | None) -> Node | None:
    if node is None:
        return None
    if node.type in _IDENTIFIER_TYPES:
        return node
    name = _field_optional(node, "declarator") or _field_optional(node, "name")
    if name is not None:
        nested = _declarator_name(name)
        if nested is not None:
            return nested
    for child in node.named_children:
        nested = _declarator_name(child)
        if nested is not None:
            return nested
    return None


def _descendant_of_type(node: Node, types: set[str]) -> Node | None:
    if node.type in types:
        return node
    for child in node.named_children:
        value = _descendant_of_type(child, types)
        if value is not None:
            return value
    return None


def _last_child_of_type(node: Node, types: set[str]) -> Node | None:
    values = [child for child in _walk_nodes(node) if child.type in types]
    return values[-1] if values else None


def _walk_nodes(node: Node) -> list[Node]:
    values = [node]
    for child in node.named_children:
        values.extend(_walk_nodes(child))
    return values


def _walk_all_nodes(node: Node) -> list[Node]:
    values = [node]
    for child in node.children:
        values.extend(_walk_all_nodes(child))
    return values


def _ancestors(node: Node, *, limit: int) -> list[Node]:
    values: list[Node] = []
    current = node.parent
    while current is not None and len(values) < limit:
        values.append(current)
        current = current.parent
    return values


def _contains(parent: Node, child: Node) -> bool:
    return parent.start_byte <= child.start_byte and child.end_byte <= parent.end_byte


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"
