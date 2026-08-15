"""File adapter bound to one leased Git worktree and execution scope."""

import os
import tempfile
from pathlib import Path

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import TaskId, WorkspaceId
from evoweave_ds.domain.validation import path_is_within_scopes
from evoweave_ds.domain.workspace_models import WorkspaceLease
from evoweave_ds.workspaces.path_policy import WorkspacePathPolicy


class GitWorktreeWorkspace:
    def __init__(
        self,
        lease: WorkspaceLease,
        *,
        max_text_bytes: int = 2_000_000,
    ) -> None:
        self._lease = lease
        self._root = Path(lease.worktree_path).resolve(strict=True)
        self._policy = WorkspacePathPolicy(
            root=self._root,
            read_scope=lease.read_scope,
            write_scope=lease.write_scope,
        )
        self._max_text_bytes = max_text_bytes

    @property
    def workspace_id(self) -> WorkspaceId:
        return self._lease.workspace_id

    @property
    def task_id(self) -> TaskId:
        return self._lease.task_id

    @property
    def root(self) -> Path:
        return self._root

    @property
    def lease(self) -> WorkspaceLease:
        return self._lease

    @property
    def path_policy(self) -> WorkspacePathPolicy:
        return self._policy

    def read_text(self, path: str) -> str:
        normalized, target = self._policy.normalize_read(path)
        size = target.stat().st_size
        if size > self._max_text_bytes:
            raise DomainError(
                ErrorCode.WORKSPACE_ACCESS_DENIED,
                f"文件超过文本读取上限：{normalized}",
            )
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_ACCESS_DENIED,
                f"文件不是 UTF-8 文本：{normalized}",
            ) from exc

    def write_text(self, path: str, content: str) -> None:
        encoded = content.encode("utf-8")
        if len(encoded) > self._max_text_bytes:
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "写入内容超过文本上限")
        _, target = self._policy.normalize_write(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=".evoweave_ds-write-",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            _, verified_target = self._policy.normalize_write(path)
            if verified_target != target:
                raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "写入目标在操作期间改变")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def list_paths(self, prefix: str | None = None) -> tuple[str, ...]:
        normalized_prefix = (
            self._policy.normalize_list_prefix(prefix) if prefix is not None else None
        )
        paths: list[str] = []
        for candidate in self._root.rglob("*"):
            relative = candidate.relative_to(self._root).as_posix()
            if relative == ".git" or relative.startswith(".git/"):
                continue
            if candidate.is_dir():
                if candidate.is_symlink() or candidate.is_junction():
                    raise DomainError(
                        ErrorCode.WORKSPACE_ACCESS_DENIED,
                        "工作区包含符号链接或目录联接",
                    )
                continue
            if normalized_prefix is not None and not path_is_within_scopes(
                relative, (normalized_prefix,)
            ):
                continue
            if not path_is_within_scopes(relative, self._lease.read_scope):
                continue
            self._policy.assert_readable_path(relative)
            paths.append(relative)
        return tuple(sorted(paths))
