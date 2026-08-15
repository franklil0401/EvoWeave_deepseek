"""Normalize repository paths and reject scope or link-based escapes."""

from pathlib import Path

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.validation import path_is_within_scopes, validate_repository_path


class WorkspacePathPolicy:
    def __init__(
        self,
        *,
        root: Path | str,
        read_scope: tuple[str, ...],
        write_scope: tuple[str, ...],
    ) -> None:
        self._root = Path(root).resolve(strict=True)
        self._read_scope = read_scope
        self._write_scope = write_scope
        if _is_link_or_junction(self._root):
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "工作区根目录不能是链接")

    @property
    def root(self) -> Path:
        return self._root

    def normalize_read(self, path: str) -> tuple[str, Path]:
        normalized = self._normalize(path)
        self._assert_scope(normalized, self._read_scope, "读取")
        candidate = self._root.joinpath(*normalized.split("/"))
        self._assert_components_safe(candidate)
        if not candidate.is_file():
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, f"文件不存在：{normalized}")
        return normalized, candidate

    def normalize_write(self, path: str) -> tuple[str, Path]:
        normalized = self._normalize(path)
        self._assert_scope(normalized, self._write_scope, "写入")
        candidate = self._root.joinpath(*normalized.split("/"))
        self._assert_components_safe(candidate)
        if candidate.exists() and not candidate.is_file():
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "写入目标不是普通文件")
        return normalized, candidate

    def normalize_list_prefix(self, prefix: str) -> str:
        normalized = self._normalize(prefix)
        self._assert_scope(normalized, self._read_scope, "列举")
        candidate = self._root.joinpath(*normalized.split("/"))
        self._assert_components_safe(candidate)
        return normalized

    def assert_readable_path(self, path: str) -> None:
        normalized = self._normalize(path)
        self._assert_scope(normalized, self._read_scope, "读取")
        self._assert_components_safe(self._root.joinpath(*normalized.split("/")))

    def assert_writable_path(self, path: str) -> None:
        normalized = self._normalize(path)
        self._assert_scope(normalized, self._write_scope, "写入")
        self._assert_components_safe(self._root.joinpath(*normalized.split("/")))

    @staticmethod
    def _normalize(path: str) -> str:
        try:
            return validate_repository_path(path)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_ACCESS_DENIED,
                "仓库路径不合法或试图逃逸工作区",
            ) from exc

    @staticmethod
    def _assert_scope(path: str, scopes: tuple[str, ...], operation: str) -> None:
        if not path_is_within_scopes(path, scopes):
            raise DomainError(
                ErrorCode.WORKSPACE_ACCESS_DENIED,
                f"{operation}路径超出授权范围：{path}",
            )

    def _assert_components_safe(self, candidate: Path) -> None:
        current = self._root
        relative_parts = candidate.relative_to(self._root).parts
        for part in relative_parts:
            current = current / part
            if _is_link_or_junction(current):
                raise DomainError(
                    ErrorCode.WORKSPACE_ACCESS_DENIED,
                    "路径包含符号链接或目录联接",
                )
        existing = candidate if candidate.exists() else candidate.parent
        while not existing.exists() and existing != self._root:
            existing = existing.parent
        resolved = existing.resolve(strict=True)
        if not resolved.is_relative_to(self._root):
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "路径解析后逃逸工作区")


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()
