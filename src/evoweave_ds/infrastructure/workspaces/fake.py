"""Pure in-memory workspace with deterministic read/write scope checks."""

from collections.abc import Mapping

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import TaskId, WorkspaceId
from evoweave_ds.domain.validation import (
    path_is_within_scopes,
    validate_repository_path,
    validate_scope_subset,
)


class FakeWorkspace:
    def __init__(
        self,
        *,
        task_id: TaskId,
        files: Mapping[str, str] | None = None,
        read_scope: tuple[str, ...] = (),
        write_scope: tuple[str, ...] = (),
    ) -> None:
        self._workspace_id = WorkspaceId.new()
        self._task_id = task_id
        self._read_scope = tuple(validate_repository_path(path) for path in read_scope)
        self._write_scope = tuple(validate_repository_path(path) for path in write_scope)
        validate_scope_subset(
            self._write_scope,
            self._read_scope,
            child_name="write_scope",
            parent_name="read_scope",
        )
        self._files = {
            validate_repository_path(path): content for path, content in (files or {}).items()
        }

    @property
    def workspace_id(self) -> WorkspaceId:
        return self._workspace_id

    @property
    def task_id(self) -> TaskId:
        return self._task_id

    def read_text(self, path: str) -> str:
        normalized = self._normalize_requested_path(path)
        self._assert_allowed(normalized, self._read_scope, "读取")
        try:
            return self._files[normalized]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_ACCESS_DENIED,
                f"文件不存在：{normalized}",
            ) from exc

    def write_text(self, path: str, content: str) -> None:
        normalized = self._normalize_requested_path(path)
        self._assert_allowed(normalized, self._write_scope, "写入")
        self._files[normalized] = content

    def list_paths(self, prefix: str | None = None) -> tuple[str, ...]:
        if prefix is None:
            return tuple(
                sorted(
                    path for path in self._files if path_is_within_scopes(path, self._read_scope)
                )
            )
        normalized = self._normalize_requested_path(prefix)
        self._assert_allowed(normalized, self._read_scope, "列举")
        return tuple(
            sorted(
                path
                for path in self._files
                if path_is_within_scopes(path, (normalized,))
                and path_is_within_scopes(path, self._read_scope)
            )
        )

    @staticmethod
    def _assert_allowed(path: str, scopes: tuple[str, ...], operation: str) -> None:
        if not path_is_within_scopes(path, scopes):
            raise DomainError(
                ErrorCode.WORKSPACE_ACCESS_DENIED,
                f"{operation}路径超出授权范围：{path}",
            )

    @staticmethod
    def _normalize_requested_path(path: str) -> str:
        try:
            return validate_repository_path(path)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_ACCESS_DENIED,
                "仓库路径不合法或试图逃逸工作区",
            ) from exc


class FakeWorkspaceProvider:
    def __init__(self, workspaces: Mapping[TaskId, FakeWorkspace]) -> None:
        self._workspaces = dict(workspaces)

    def for_execution(self, execution_spec: AgentExecutionSpec) -> FakeWorkspace:
        try:
            workspace = self._workspaces[execution_spec.task_id]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_ACCESS_DENIED,
                f"任务没有工作区：{execution_spec.task_id}",
            ) from exc
        if workspace.task_id != execution_spec.task_id:
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "工作区与任务不匹配")
        return workspace
