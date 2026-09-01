"""Typed records shared by the parser, index, router, and provider tool."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SemanticAction(StrEnum):
    SYMBOLS = "symbols"
    WORKSPACE_SYMBOLS = "workspace_symbols"
    DEFINITION = "definition"
    REFERENCES = "references"
    DIAGNOSTICS = "diagnostics"


class QueryPrecision(StrEnum):
    AUTO = "auto"
    TEXT = "text"
    SYNTAX = "syntax"
    PROJECT = "project"
    COMPILER = "compiler"


class QueryStatus(StrEnum):
    EXACT = "exact"
    AMBIGUOUS = "ambiguous"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class Confidence(StrEnum):
    EXACT = "exact"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


SymbolKind = Literal[
    "function",
    "method",
    "class",
    "struct",
    "interface",
    "variable",
    "constant",
    "module",
    "namespace",
    "field",
    "parameter",
]
ReferenceKind = Literal[
    "call",
    "read",
    "write",
    "type",
    "import",
    "inheritance",
    "unknown",
]


class IndexedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    language: str
    snapshot_id: str
    content_hash: str
    indexed_at_workspace_revision: int = Field(ge=0)
    parse_error_count: int = Field(ge=0)


class SymbolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol_id: str
    name: str
    kind: SymbolKind
    path: str
    snapshot_id: str
    start_line: int = Field(ge=1)
    start_column: int = Field(ge=1)
    end_line: int = Field(ge=1)
    end_column: int = Field(ge=1)
    scope_id: str | None = None
    qualified_name: str | None = None
    signature: str | None = None
    documentation: str | None = None
    exported: bool = False


class ReferenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_id: str
    name: str
    path: str
    snapshot_id: str
    line: int = Field(ge=1)
    column: int = Field(ge=1)
    scope_id: str | None = None
    context_kind: ReferenceKind


class ImportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    imported_name: str
    local_alias: str | None = None
    target_module: str
    line: int = Field(ge=1)
    snapshot_id: str


class ScopeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_id: str
    parent_scope_id: str | None = None
    kind: str
    name: str | None = None
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    snapshot_id: str


class SemanticDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    line: int | None = Field(default=None, ge=1)
    column: int | None = Field(default=None, ge=1)
    severity: Literal["error", "warning", "info"]
    source: str
    code: str | None = None
    message: str
    workspace_revision: int = Field(ge=0)


class SemanticLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    snapshot_id: str | None = None
    name: str | None = None
    kind: str | None = None
    start_line: int = Field(ge=1)
    start_column: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    end_column: int | None = Field(default=None, ge=1)
    confidence: Confidence
    source: Literal["tree_sitter", "text_search", "compiler"]
    reason: str
    signature: str | None = None
    documentation: str | None = None
    context_kind: ReferenceKind | None = None
    diagnostic: SemanticDiagnostic | None = None


class SemanticQueryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: SemanticAction = Field(
        description=(
            "symbols lists one file; workspace_symbols searches names; definition and "
            "references resolve a symbol or source position; diagnostics checks syntax"
        )
    )
    path: str | None = Field(
        default=None,
        description=(
            "Workspace-relative file. Required for symbols and for position-based definition "
            "or references; optional for diagnostics."
        ),
    )
    line: int | None = Field(
        default=None,
        ge=1,
        description="One-based source line; use together with path and column.",
    )
    column: int | None = Field(
        default=None,
        ge=1,
        description="One-based source column; use together with path and line.",
    )
    symbol: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Exact symbol name for definition or references. This is the non-positional "
            "alternative to path+line+column."
        ),
    )
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Name fragment used only by workspace_symbols.",
    )
    precision: QueryPrecision = QueryPrecision.AUTO

    @model_validator(mode="after")
    def validate_action_inputs(self) -> SemanticQueryArguments:
        if self.action == SemanticAction.SYMBOLS and not self.path:
            raise ValueError("symbols requires path")
        if self.action == SemanticAction.WORKSPACE_SYMBOLS and not (self.query or self.symbol):
            raise ValueError("workspace_symbols requires query or symbol")
        if self.action in {SemanticAction.DEFINITION, SemanticAction.REFERENCES}:
            has_position = (
                self.path is not None and self.line is not None and self.column is not None
            )
            if not has_position and not self.symbol:
                raise ValueError(f"{self.action.value} requires symbol or path, line, and column")
        return self


class SemanticQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: SemanticAction
    status: QueryStatus
    results: list[SemanticLocation] = Field(default_factory=list, max_length=200)
    confidence: Confidence
    evidence_refs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    workspace_revision: int = Field(ge=0)
    index_revision: int = Field(ge=0)
    snapshot_ids: dict[str, str] = Field(default_factory=dict)
