"""Classify committed repository files without reading the mutable worktree."""

from hashlib import sha256
from pathlib import PurePosixPath

from evoweave_ds.domain.repository_models import (
    RepositoryBlob,
    RepositoryFile,
    RepositoryFileKind,
)
from evoweave_ds.repository.git_inspector import GitInspector

_CONFIG_NAMES = {
    ".editorconfig",
    ".gitignore",
    ".pre-commit-config.yaml",
    "mypy.ini",
    "pytest.ini",
    "ruff.toml",
    "tox.ini",
}
_BUILD_NAMES = {
    "dockerfile",
    "makefile",
    "pdm.lock",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "uv.lock",
}
_TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class FileInventoryBuilder:
    def build(
        self,
        inspector: GitInspector,
        blobs: tuple[RepositoryBlob, ...] | None = None,
    ) -> tuple[RepositoryFile, ...]:
        items: list[RepositoryFile] = []
        for blob in blobs if blobs is not None else inspector.list_blobs():
            if not blob.is_regular_file:
                continue
            data = inspector.read_bytes(blob.path)
            kind = classify_file(blob.path)
            language = detect_language(blob.path)
            line_count = count_lines(data) if is_probably_text(blob.path, data) else 0
            items.append(
                RepositoryFile(
                    path=blob.path,
                    object_id=blob.object_id,
                    sha256=sha256(data).hexdigest(),
                    kind=kind,
                    language=language,
                    size_bytes=len(data),
                    line_count=line_count,
                    module_name=python_module_name(blob.path)
                    if blob.path.endswith(".py")
                    else None,
                )
            )
        return tuple(sorted(items, key=lambda item: item.path))


def classify_file(path: str) -> RepositoryFileKind:
    pure_path = PurePosixPath(path)
    name = pure_path.name.lower()
    parts = tuple(part.lower() for part in pure_path.parts)
    if path.startswith(".github/workflows/") or (
        "ci" in parts and pure_path.suffix in {".yml", ".yaml"}
    ):
        return "ci"
    if "tests" in parts or "test" in parts or name.startswith("test_") or name.endswith("_test.py"):
        return "test"
    if name in _BUILD_NAMES:
        return "build"
    if name in _CONFIG_NAMES or pure_path.suffix.lower() in {
        ".ini",
        ".cfg",
        ".toml",
        ".yaml",
        ".yml",
    }:
        return "configuration"
    if pure_path.suffix.lower() == ".py":
        return "python_source"
    if pure_path.suffix.lower() in {".md", ".rst"}:
        return "documentation"
    return "other"


def detect_language(path: str) -> str | None:
    suffix = PurePosixPath(path).suffix.lower()
    return {
        ".cfg": "ini",
        ".ini": "ini",
        ".json": "json",
        ".md": "markdown",
        ".py": "python",
        ".rst": "restructuredtext",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(suffix)


def is_probably_text(path: str, data: bytes) -> bool:
    if PurePosixPath(path).suffix.lower() in _TEXT_SUFFIXES:
        return True
    return b"\0" not in data[:8_192]


def count_lines(data: bytes) -> int:
    if not data:
        return 0
    return len(data.splitlines())


def python_module_name(path: str) -> str:
    pure_path = PurePosixPath(path)
    parts = list(pure_path.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or "__root__"
