"""Provider-facing read-only semantic query tool."""

from __future__ import annotations

import json

from repo_rivet.approval.models import Capability
from repo_rivet.semantic.engine import SemanticEngine
from repo_rivet.semantic.models import SemanticQueryArguments, SemanticQueryResult
from repo_rivet.tools.base import BaseTool, ToolResult


class SemanticQueryTool(BaseTool[SemanticQueryArguments]):
    name = "semantic_query"
    description = (
        "Query the snapshot-bound lightweight code index. Supports file symbols, workspace "
        "symbols, ranked definition candidates, syntax-filtered reference candidates, and "
        "Tree-sitter syntax diagnostics for C/C++, Python, JavaScript, and TypeScript. Results "
        "include confidence and warnings; low-confidence candidates are navigation evidence, "
        "not permission to edit. Compiler diagnostics still require registered verification."
    )
    arguments_type = SemanticQueryArguments
    capabilities = frozenset({Capability.FILESYSTEM_READ})

    def __init__(self, engine: SemanticEngine) -> None:
        self.engine = engine

    def run(self, arguments: SemanticQueryArguments) -> ToolResult:
        result = self.engine.query(arguments)
        paths = list(dict.fromkeys(item.path for item in result.results))[:30]
        return ToolResult(
            ok=result.status.value != "error",
            output=json.dumps(
                _context_payload(result),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            metadata={
                "action": result.action.value,
                "status": result.status.value,
                "confidence": result.confidence.value,
                "result_count": len(result.results),
                "paths": paths,
                "snapshot_ids": result.snapshot_ids,
                "workspace_revision": result.workspace_revision,
                "index_revision": result.index_revision,
                "warning_count": len(result.warnings),
            },
        )


_MAX_CONTEXT_RESULTS = 50


def _context_payload(result: SemanticQueryResult) -> dict[str, object]:
    """Serialize useful navigation evidence without duplicating index bookkeeping."""
    selected = result.results[:_MAX_CONTEXT_RESULTS]
    locations: list[dict[str, object]] = []
    for item in selected:
        location = item.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"diagnostic", "documentation"},
        )
        locations.append(location)
    warnings = list(result.warnings)
    omitted = len(result.results) - len(selected)
    if omitted:
        warnings.append(f"{omitted} additional results omitted; narrow the query to inspect them.")
    return {
        "action": result.action.value,
        "status": result.status.value,
        "confidence": result.confidence.value,
        "results": locations,
        "warnings": warnings,
        "workspace_revision": result.workspace_revision,
    }
