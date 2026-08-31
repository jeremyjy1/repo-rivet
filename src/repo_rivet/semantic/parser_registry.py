"""Language detection and construction of offline Tree-sitter parsers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
from tree_sitter import Language, Parser, Tree


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    path: str
    language: str
    source: bytes
    tree: Tree


LanguageFactory = Callable[[], object]

_LANGUAGE_FACTORIES: dict[str, LanguageFactory] = {
    "c": tree_sitter_c.language,
    "cpp": tree_sitter_cpp.language,
    "python": tree_sitter_python.language,
    "javascript": tree_sitter_javascript.language,
    "typescript": tree_sitter_typescript.language_typescript,
    "tsx": tree_sitter_typescript.language_tsx,
}
_EXTENSIONS = {
    ".c": "c",
    ".h": "cpp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
}


class ParserRegistry:
    def __init__(self) -> None:
        self._languages: dict[str, Language] = {}

    @staticmethod
    def language_for_path(path: str | Path) -> str | None:
        return _EXTENSIONS.get(Path(path).suffix.casefold())

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(_EXTENSIONS)

    def parse(self, path: str, content: bytes) -> ParsedDocument:
        language_id = self.language_for_path(path)
        if language_id is None:
            raise ValueError(f"Unsupported semantic language: {Path(path).suffix or path}")
        language = self._languages.get(language_id)
        if language is None:
            language = Language(_LANGUAGE_FACTORIES[language_id]())
            self._languages[language_id] = language
        tree = Parser(language).parse(content)
        return ParsedDocument(path=path, language=language_id, source=content, tree=tree)
