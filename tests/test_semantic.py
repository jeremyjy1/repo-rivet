from __future__ import annotations

import json
from pathlib import Path

import pytest

from repo_rivet.editing.runtime import EditingRuntime
from repo_rivet.planning.runtime import PLANNING_TOOL_NAMES
from repo_rivet.safety.path_policy import PathPolicyError, WorkspacePathPolicy
from repo_rivet.semantic.engine import SemanticEngine
from repo_rivet.semantic.models import (
    Confidence,
    QueryStatus,
    SemanticAction,
    SemanticQueryArguments,
)
from repo_rivet.semantic.parser_registry import ParserRegistry
from repo_rivet.semantic.tools import SemanticQueryTool
from repo_rivet.subagents.policy import ScopedWorkspacePathPolicy
from repo_rivet.tools.base import ToolCall
from repo_rivet.tools.registry import create_default_registry


def _engine(workspace: Path) -> SemanticEngine:
    policy = WorkspacePathPolicy(workspace)
    editing = EditingRuntime(policy, snapshot_dir=workspace / ".session" / "snapshots")
    return SemanticEngine(
        policy,
        editing,
        index_path=workspace / ".session" / "index" / "semantic.sqlite",
    )


@pytest.mark.parametrize(
    ("path", "source", "language"),
    [
        ("sample.c", b"int main(void) { return 0; }\n", "c"),
        ("sample.cpp", b"class Item {};\n", "cpp"),
        ("sample.py", b"def work():\n    return 1\n", "python"),
        ("sample.js", b"export function work() {}\n", "javascript"),
        ("sample.ts", b"interface Item { value: number }\n", "typescript"),
        ("sample.tsx", b"const View = () => <div />;\n", "tsx"),
    ],
)
def test_parser_registry_uses_offline_language_adapters(
    path: str,
    source: bytes,
    language: str,
) -> None:
    parsed = ParserRegistry().parse(path, source)

    assert parsed.language == language
    assert not parsed.tree.root_node.has_error


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        ("sample.py", "class Worker:\n    def run(self):\n        return 1\n", {"Worker", "run"}),
        ("sample.cpp", "class Worker { public: void run() {} };\n", {"Worker", "run"}),
        ("sample.js", "export class Worker { run() {} }\n", {"Worker", "run"}),
        (
            "sample.ts",
            "export interface Worker { run(): void }\n",
            {"Worker", "run"},
        ),
    ],
)
def test_file_symbols_are_extracted_for_supported_languages(
    tmp_path: Path,
    path: str,
    source: str,
    expected: set[str],
) -> None:
    (tmp_path / path).write_text(source, encoding="utf-8")

    result = _engine(tmp_path).query(
        SemanticQueryArguments(action=SemanticAction.SYMBOLS, path=path)
    )

    assert result.status == QueryStatus.EXACT
    assert expected <= {item.name for item in result.results}
    assert result.snapshot_ids[path]
    assert all(item.source == "tree_sitter" for item in result.results)


def test_workspace_symbols_rank_exact_name_before_prefixes(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text(
        "def render():\n    pass\n\ndef renderer():\n    pass\n",
        encoding="utf-8",
    )

    result = _engine(tmp_path).query(
        SemanticQueryArguments(
            action=SemanticAction.WORKSPACE_SYMBOLS,
            query="render",
        )
    )

    assert [item.name for item in result.results[:2]] == ["render", "renderer"]
    assert result.results[0].confidence == Confidence.EXACT
    assert result.results[1].confidence == Confidence.HIGH


def test_workspace_scan_skips_generated_and_local_state_directories(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("def visible():\n    pass\n", encoding="utf-8")
    for directory in (".local", "dist", "web_dist"):
        target = tmp_path / directory
        target.mkdir()
        (target / "generated.py").write_text(
            "def hidden_generated():\n    pass\n",
            encoding="utf-8",
        )

    result = _engine(tmp_path).query(
        SemanticQueryArguments(
            action=SemanticAction.WORKSPACE_SYMBOLS,
            query="hidden_generated",
        )
    )

    assert result.results == []


def test_scoped_semantic_engine_only_indexes_delegated_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    excluded = tmp_path / "outside"
    allowed.mkdir()
    excluded.mkdir()
    (allowed / "inside.py").write_text("def inside():\n    pass\n", encoding="utf-8")
    (excluded / "outside.py").write_text("def outside():\n    pass\n", encoding="utf-8")
    policy = ScopedWorkspacePathPolicy(tmp_path, allowed_paths=["allowed"])
    editing = EditingRuntime(policy, snapshot_dir=tmp_path / ".session" / "snapshots")
    engine = SemanticEngine(policy, editing, scan_roots=["allowed"])

    inside = engine.query(
        SemanticQueryArguments(action=SemanticAction.WORKSPACE_SYMBOLS, query="inside")
    )
    outside = engine.query(
        SemanticQueryArguments(action=SemanticAction.WORKSPACE_SYMBOLS, query="outside")
    )

    assert [item.path for item in inside.results] == ["allowed/inside.py"]
    assert outside.results == []


def test_definition_uses_position_and_returns_snapshot_bound_candidate(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.py").write_text(
        "def calculate(value):\n    return value + 1\n\nresult = calculate(2)\n",
        encoding="utf-8",
    )

    result = _engine(tmp_path).query(
        SemanticQueryArguments(
            action=SemanticAction.DEFINITION,
            path="sample.py",
            line=4,
            column=12,
        )
    )

    assert result.status == QueryStatus.EXACT
    assert result.results[0].name == "calculate"
    assert result.results[0].start_line == 1
    assert result.results[0].snapshot_id == result.snapshot_ids["sample.py"]


def test_definition_columns_are_character_based_for_unicode_identifiers(
    tmp_path: Path,
) -> None:
    (tmp_path / "unicode.py").write_text(
        "def 计算():\n    return 1\n\n结果 = 计算()\n",
        encoding="utf-8",
    )

    result = _engine(tmp_path).query(
        SemanticQueryArguments(
            action=SemanticAction.DEFINITION,
            path="unicode.py",
            line=4,
            column=6,
        )
    )

    assert result.status == QueryStatus.EXACT
    assert result.results[0].name == "计算"
    assert result.results[0].start_column == 5


@pytest.mark.parametrize(
    ("library_path", "library_source", "consumer_path", "consumer_source", "line", "column"),
    [
        (
            "library.py",
            "class Worker:\n    pass\n",
            "consumer.py",
            "from library import Worker as Alias\nvalue = Alias()\n",
            2,
            10,
        ),
        (
            "library.js",
            "export class Worker {}\n",
            "consumer.js",
            "import { Worker as Alias } from './library.js';\nconst value = new Alias();\n",
            2,
            20,
        ),
    ],
)
def test_import_aliases_rank_project_definition_candidates(
    tmp_path: Path,
    library_path: str,
    library_source: str,
    consumer_path: str,
    consumer_source: str,
    line: int,
    column: int,
) -> None:
    (tmp_path / library_path).write_text(library_source, encoding="utf-8")
    (tmp_path / consumer_path).write_text(consumer_source, encoding="utf-8")

    result = _engine(tmp_path).query(
        SemanticQueryArguments(
            action=SemanticAction.DEFINITION,
            path=consumer_path,
            line=line,
            column=column,
            precision="project",
        )
    )

    assert result.status == QueryStatus.EXACT
    assert result.results[0].path == library_path
    assert result.results[0].name == "Worker"
    assert result.results[0].confidence == Confidence.HIGH


def test_references_keep_syntax_and_text_precision_distinct(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text(
        "def calculate():\n    return 1\n\nvalue = calculate()\n# calculate in a comment\n",
        encoding="utf-8",
    )

    engine = _engine(tmp_path)
    result = engine.query(
        SemanticQueryArguments(action=SemanticAction.REFERENCES, symbol="calculate")
    )
    text_result = engine.query(
        SemanticQueryArguments(
            action=SemanticAction.REFERENCES,
            symbol="calculate",
            precision="text",
        )
    )

    syntax = [item for item in result.results if item.source == "tree_sitter"]
    text = [item for item in text_result.results if item.source == "text_search"]
    assert any(item.start_line == 4 for item in syntax)
    assert all(item.start_line != 5 for item in syntax)
    assert any(item.start_line == 5 for item in text)
    assert all(item.confidence == Confidence.LOW for item in text)
    assert any("comments" in warning for warning in text_result.warnings)


def test_syntax_diagnostics_report_parser_errors_without_execution(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

    result = _engine(tmp_path).query(
        SemanticQueryArguments(action=SemanticAction.DIAGNOSTICS, path="broken.py")
    )
    compiler = _engine(tmp_path).query(
        SemanticQueryArguments(
            action=SemanticAction.DIAGNOSTICS,
            path="broken.py",
            precision="compiler",
        )
    )

    assert result.status == QueryStatus.EXACT
    assert result.results
    assert all(item.kind == "diagnostic" for item in result.results)
    assert all(
        item.diagnostic and item.diagnostic.source == "tree_sitter" for item in result.results
    )
    assert any("type correctness" in warning for warning in result.warnings)
    assert compiler.status == QueryStatus.PARTIAL
    assert compiler.confidence == Confidence.LOW
    assert any("registered verification" in warning for warning in compiler.warnings)


def test_incremental_index_only_advances_when_file_content_changes(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text("def first():\n    pass\n", encoding="utf-8")
    engine = _engine(tmp_path)
    request = SemanticQueryArguments(action=SemanticAction.SYMBOLS, path="sample.py")

    first = engine.query(request)
    unchanged = engine.query(request)
    path.write_text("def second():\n    pass\n", encoding="utf-8")
    changed = engine.query(request)

    assert unchanged.index_revision == first.index_revision
    assert changed.index_revision > unchanged.index_revision
    assert {item.name for item in changed.results} == {"second"}


def test_persistent_index_is_reused_by_a_new_engine(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("def stable():\n    pass\n", encoding="utf-8")
    request = SemanticQueryArguments(action=SemanticAction.SYMBOLS, path="sample.py")
    first = _engine(tmp_path).query(request)

    second = _engine(tmp_path).query(request)

    assert second.index_revision == first.index_revision
    assert second.snapshot_ids == first.snapshot_ids


def test_unsupported_language_returns_explicit_fallback_guidance(tmp_path: Path) -> None:
    (tmp_path / "notes.rb").write_text("puts 'hello'\n", encoding="utf-8")

    result = _engine(tmp_path).query(
        SemanticQueryArguments(action=SemanticAction.SYMBOLS, path="notes.rb")
    )

    assert result.status == QueryStatus.UNSUPPORTED
    assert result.results == []
    assert "search_text" in result.warnings[0]


def test_semantic_query_obeys_workspace_confinement(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("def hidden():\n    pass\n", encoding="utf-8")

    with pytest.raises(PathPolicyError):
        _engine(tmp_path).query(
            SemanticQueryArguments(action=SemanticAction.SYMBOLS, path=str(outside))
        )
    with pytest.raises(PathPolicyError):
        _engine(tmp_path).query(
            SemanticQueryArguments(
                action=SemanticAction.DEFINITION,
                path=str(outside),
                symbol="hidden",
            )
        )


def test_semantic_tool_is_read_only_and_available_while_planning(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("def visible():\n    pass\n", encoding="utf-8")
    registry = create_default_registry(
        tmp_path,
        snapshot_dir=tmp_path / ".session" / "snapshots",
        semantic_index_path=tmp_path / ".session" / "index" / "semantic.sqlite",
    )

    result = registry.execute(
        ToolCall(
            id="call-semantic",
            name="semantic_query",
            arguments={"action": "symbols", "path": "sample.py"},
        )
    )

    assert result.ok
    assert not registry.is_state_changing("semantic_query")
    assert "semantic_query" in PLANNING_TOOL_NAMES
    tool = next(
        schema for schema in registry.schemas() if schema["function"]["name"] == "semantic_query"
    )
    assert "Compiler diagnostics" in tool["function"]["description"]
    assert "Do not pass query to definition or references" in tool["function"]["description"]
    properties = tool["function"]["parameters"]["properties"]
    assert "used only by workspace_symbols" in properties["query"]["description"]
    assert "Exact symbol name" in properties["symbol"]["description"]


def test_semantic_tool_returns_compact_context_without_unrelated_snapshots(
    tmp_path: Path,
) -> None:
    (tmp_path / "clean.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    tool = SemanticQueryTool(_engine(tmp_path))

    clean = tool.execute({"action": "diagnostics"})

    assert clean.ok
    assert clean.metadata and clean.metadata["snapshot_ids"] == {}
    assert json.loads(clean.output)["results"] == []

    (tmp_path / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    broken = tool.execute({"action": "diagnostics", "path": "broken.py"})
    payload = json.loads(broken.output)

    assert broken.ok
    assert payload["results"]
    assert "diagnostic" not in payload["results"][0]
    assert "documentation" not in payload["results"][0]


def test_semantic_tool_rejects_invalid_action_arguments(tmp_path: Path) -> None:
    tool = SemanticQueryTool(_engine(tmp_path))

    result = tool.execute({"action": "definition"})

    assert not result.ok
    assert result.error is not None
