"""Extract Python symbols and imports with the standard-library AST."""

import ast
from dataclasses import dataclass

from evoweave_ds.domain.repository_models import (
    PythonImport,
    PythonSymbol,
    PythonSymbolKind,
    RepositoryParseIssue,
)
from evoweave_ds.repository.evidence_builder import deterministic_evidence_id
from evoweave_ds.repository.file_inventory import python_module_name


@dataclass(frozen=True, slots=True)
class PythonAnalysis:
    symbols: tuple[PythonSymbol, ...]
    imports: tuple[PythonImport, ...]
    issues: tuple[RepositoryParseIssue, ...]


class PythonSymbolExtractor:
    def analyze(self, *, path: str, content: bytes, base_commit: str) -> PythonAnalysis:
        module_name = python_module_name(path)
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            return PythonAnalysis(
                symbols=(),
                imports=(),
                issues=(
                    RepositoryParseIssue(
                        path=path,
                        line=1,
                        message="Python 文件不是有效 UTF-8 文本",
                        evidence_id=deterministic_evidence_id(
                            base_commit, path, "decode", exc.start
                        ),
                    ),
                ),
            )
        try:
            tree = ast.parse(text, filename=path, type_comments=True)
        except SyntaxError as exc:
            return PythonAnalysis(
                symbols=(),
                imports=(),
                issues=(
                    RepositoryParseIssue(
                        path=path,
                        line=exc.lineno,
                        message=f"Python 语法错误：{exc.msg}",
                        evidence_id=deterministic_evidence_id(
                            base_commit, path, "syntax", exc.lineno, exc.msg
                        ),
                    ),
                ),
            )

        line_count = max(1, len(text.splitlines()))
        symbols: list[PythonSymbol] = [
            PythonSymbol(
                path=path,
                module_name=module_name,
                qualified_name=module_name,
                name=module_name.rsplit(".", 1)[-1],
                kind="module",
                line_start=1,
                line_end=line_count,
                evidence_id=deterministic_evidence_id(
                    base_commit, path, 1, line_count, module_name
                ),
            )
        ]
        visitor = _DefinitionVisitor(path, module_name, base_commit)
        visitor.visit(tree)
        symbols.extend(visitor.symbols)
        imports = _extract_imports(
            tree=tree,
            path=path,
            module_name=module_name,
            base_commit=base_commit,
        )
        return PythonAnalysis(
            symbols=tuple(sorted(symbols, key=_symbol_sort_key)),
            imports=tuple(sorted(imports, key=_import_sort_key)),
            issues=(),
        )


class _DefinitionVisitor(ast.NodeVisitor):
    def __init__(self, path: str, module_name: str, base_commit: str) -> None:
        self._path = path
        self._module_name = module_name
        self._base_commit = base_commit
        self._scope: list[tuple[str, str]] = []
        self.symbols: list[PythonSymbol] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node, "class")
        self._scope.append((node.name, "class"))
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind: PythonSymbolKind = (
            "method" if any(scope_kind == "class" for _, scope_kind in self._scope) else "function"
        )
        self._add(node, kind)
        self._scope.append((node.name, "function"))
        self.generic_visit(node)
        self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind: PythonSymbolKind = (
            "method" if any(scope_kind == "class" for _, scope_kind in self._scope) else "function"
        )
        self._add(node, kind)
        self._scope.append((node.name, "function"))
        self.generic_visit(node)
        self._scope.pop()

    def _add(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: PythonSymbolKind,
    ) -> None:
        qualified_parts = [self._module_name, *(name for name, _ in self._scope), node.name]
        qualified_name = ".".join(qualified_parts)
        line_end = node.end_lineno or node.lineno
        self.symbols.append(
            PythonSymbol(
                path=self._path,
                module_name=self._module_name,
                qualified_name=qualified_name,
                name=node.name,
                kind=kind,
                line_start=node.lineno,
                line_end=line_end,
                evidence_id=deterministic_evidence_id(
                    self._base_commit,
                    self._path,
                    node.lineno,
                    line_end,
                    qualified_name,
                ),
            )
        )


def _extract_imports(
    *,
    tree: ast.AST,
    path: str,
    module_name: str,
    base_commit: str,
) -> list[PythonImport]:
    imports: list[PythonImport] = []
    is_package = path.endswith("/__init__.py") or path == "__init__.py"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    _make_import(
                        path=path,
                        module_name=module_name,
                        imported_module=alias.name,
                        imported_name=None,
                        level=0,
                        line=node.lineno,
                        base_commit=base_commit,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            imported_module = _resolve_from_module(
                importer_module=module_name,
                imported_module=node.module,
                level=node.level,
                is_package=is_package,
            )
            if not imported_module:
                continue
            for alias in node.names:
                imports.append(
                    _make_import(
                        path=path,
                        module_name=module_name,
                        imported_module=imported_module,
                        imported_name=alias.name,
                        level=node.level,
                        line=node.lineno,
                        base_commit=base_commit,
                    )
                )
    return imports


def _make_import(
    *,
    path: str,
    module_name: str,
    imported_module: str,
    imported_name: str | None,
    level: int,
    line: int,
    base_commit: str,
) -> PythonImport:
    return PythonImport(
        path=path,
        importer_module=module_name,
        imported_module=imported_module,
        imported_name=imported_name,
        level=level,
        line=line,
        evidence_id=deterministic_evidence_id(
            base_commit,
            path,
            line,
            imported_module,
            imported_name,
        ),
    )


def _resolve_from_module(
    *,
    importer_module: str,
    imported_module: str | None,
    level: int,
    is_package: bool,
) -> str:
    if level == 0:
        return imported_module or ""
    package_parts = importer_module.split(".")
    if not is_package:
        package_parts = package_parts[:-1]
    remove_count = level - 1
    if remove_count > len(package_parts):
        return imported_module or ""
    base = package_parts[: len(package_parts) - remove_count]
    if imported_module:
        base.extend(imported_module.split("."))
    return ".".join(base)


def _symbol_sort_key(item: PythonSymbol) -> tuple[str, int, str]:
    return (item.path, item.line_start, item.qualified_name)


def _import_sort_key(item: PythonImport) -> tuple[str, int, str, str]:
    return (item.path, item.line, item.imported_module, item.imported_name or "")
