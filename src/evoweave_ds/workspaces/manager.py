"""Create, persist, recover, and release isolated worktree leases."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.base import utc_now
from evoweave_ds.domain.enums import WorkspaceAccessMode, WorkspaceLeaseStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import WorkspaceId
from evoweave_ds.domain.workspace_models import WorkspaceLease
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.workspaces.git_worktree import GitWorktreeController
from evoweave_ds.workspaces.local_workspace import GitWorktreeWorkspace
from evoweave_ds.workspaces.state_store import JsonWorkspaceLeaseStore


class WorkspaceManager:
    def __init__(
        self,
        *,
        repository_root: Path | str,
        worktree_root: Path | str,
        lease_store: JsonWorkspaceLeaseStore,
    ) -> None:
        self._inspector = GitInspector(repository_root)
        self._controller = GitWorktreeController(
            self._inspector.repository_root,
            worktree_root,
        )
        self._store = lease_store
        self._lock = RLock()

    def create(self, execution_spec: AgentExecutionSpec) -> WorkspaceLease:
        with self._lock:
            return self._create(execution_spec)

    def _create(self, execution_spec: AgentExecutionSpec) -> WorkspaceLease:
        if (
            execution_spec.base_commit
            != GitInspector(
                self._inspector.repository_root,
                execution_spec.base_commit,
            ).base_commit
        ):
            raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "无法解析执行规格基线")
        workspace_id = WorkspaceId.new()
        access_mode = (
            WorkspaceAccessMode.READ_WRITE
            if execution_spec.write_scope
            else WorkspaceAccessMode.READ_ONLY
        )
        branch_name = _branch_name(execution_spec, workspace_id)
        path = self._controller.path_for(workspace_id)
        lease = WorkspaceLease(
            workspace_id=workspace_id,
            run_id=execution_spec.run_id,
            task_id=execution_spec.task_id,
            agent_id=execution_spec.agent_id,
            execution_spec_id=execution_spec.spec_id,
            execution_spec_version=execution_spec.version,
            repository_root=str(self._inspector.repository_root),
            worktree_path=str(path),
            branch_name=branch_name,
            base_commit=execution_spec.base_commit,
            access_mode=access_mode,
            read_scope=execution_spec.read_scope,
            write_scope=execution_spec.write_scope,
        )
        self._store.save(lease)
        try:
            self._controller.create(
                workspace_id=workspace_id,
                branch_name=branch_name,
                base_commit=execution_spec.base_commit,
            )
        except DomainError as exc:
            failed = lease.model_copy(
                update={
                    "status": WorkspaceLeaseStatus.FAILED,
                    "failure_reason": exc.message,
                    "updated_at": utc_now(),
                    "version": lease.version + 1,
                }
            )
            self._store.save(failed)
            raise
        active = lease.model_copy(
            update={
                "status": WorkspaceLeaseStatus.ACTIVE,
                "updated_at": utc_now(),
                "version": lease.version + 1,
            }
        )
        self._store.save(active)
        return active

    def open(
        self, lease: WorkspaceLease, execution_spec: AgentExecutionSpec
    ) -> GitWorktreeWorkspace:
        persisted = self._store.get(lease.workspace_id)
        if persisted != lease:
            lease = persisted
        if lease.status is not WorkspaceLeaseStatus.ACTIVE:
            raise DomainError(ErrorCode.WORKSPACE_STATE_INVALID, "工作区租约不是 active 状态")
        _validate_binding(lease, execution_spec)
        if not self._controller.is_registered(lease.worktree_path):
            raise DomainError(ErrorCode.WORKSPACE_STATE_INVALID, "worktree 未在 Git 中注册")
        if self._controller.head(lease.worktree_path) != lease.base_commit:
            raise DomainError(ErrorCode.WORKSPACE_STATE_INVALID, "worktree HEAD 已偏离租约基线")
        return GitWorktreeWorkspace(lease)

    def for_execution(self, execution_spec: AgentExecutionSpec) -> GitWorktreeWorkspace:
        matches = [
            lease
            for lease in self._store.list_all()
            if lease.execution_spec_id == execution_spec.spec_id
            and lease.execution_spec_version == execution_spec.version
            and lease.status is WorkspaceLeaseStatus.ACTIVE
        ]
        if len(matches) != 1:
            raise DomainError(
                ErrorCode.WORKSPACE_LEASE_NOT_FOUND,
                "执行规格必须恰好绑定一个 active 工作区",
            )
        return self.open(matches[0], execution_spec)

    @contextmanager
    def session(
        self,
        execution_spec: AgentExecutionSpec,
    ) -> Iterator[tuple[WorkspaceLease, GitWorktreeWorkspace]]:
        lease = self.create(execution_spec)
        try:
            yield lease, self.open(lease, execution_spec)
        finally:
            self.release(lease.workspace_id)

    def release(self, workspace_id: WorkspaceId) -> WorkspaceLease:
        with self._lock:
            return self._release(workspace_id)

    def _release(self, workspace_id: WorkspaceId) -> WorkspaceLease:
        lease = self._store.get(workspace_id)
        if lease.status is WorkspaceLeaseStatus.RELEASED:
            return lease
        if lease.status not in {
            WorkspaceLeaseStatus.ACTIVE,
            WorkspaceLeaseStatus.ORPHANED,
            WorkspaceLeaseStatus.RELEASING,
        }:
            raise DomainError(ErrorCode.WORKSPACE_STATE_INVALID, "当前租约状态不能回收")
        releasing = lease.model_copy(
            update={
                "status": WorkspaceLeaseStatus.RELEASING,
                "updated_at": utc_now(),
                "version": lease.version + 1,
            }
        )
        self._store.save(releasing)
        self._controller.remove(
            path=releasing.worktree_path,
            branch_name=releasing.branch_name,
        )
        released = releasing.model_copy(
            update={
                "status": WorkspaceLeaseStatus.RELEASED,
                "updated_at": utc_now(),
                "version": releasing.version + 1,
            }
        )
        self._store.save(released)
        return released

    def recover(self) -> tuple[WorkspaceLease, ...]:
        with self._lock:
            return self._recover()

    def _recover(self) -> tuple[WorkspaceLease, ...]:
        recovered: list[WorkspaceLease] = []
        for lease in self._store.list_all():
            if lease.status is WorkspaceLeaseStatus.CREATING:
                recovered.append(self._recover_creating(lease))
            elif lease.status is WorkspaceLeaseStatus.RELEASING:
                recovered.append(self.release(lease.workspace_id))
            elif lease.status is WorkspaceLeaseStatus.ACTIVE:
                if self._controller.is_registered(lease.worktree_path):
                    recovered.append(lease)
                else:
                    orphaned = lease.model_copy(
                        update={
                            "status": WorkspaceLeaseStatus.ORPHANED,
                            "updated_at": utc_now(),
                            "version": lease.version + 1,
                        }
                    )
                    self._store.save(orphaned)
                    recovered.append(orphaned)
        return tuple(recovered)

    def _recover_creating(self, lease: WorkspaceLease) -> WorkspaceLease:
        if self._controller.is_registered(lease.worktree_path):
            active = lease.model_copy(
                update={
                    "status": WorkspaceLeaseStatus.ACTIVE,
                    "updated_at": utc_now(),
                    "version": lease.version + 1,
                }
            )
            self._store.save(active)
            return active
        failed = lease.model_copy(
            update={
                "status": WorkspaceLeaseStatus.FAILED,
                "failure_reason": "创建过程在 worktree 注册前中断",
                "updated_at": utc_now(),
                "version": lease.version + 1,
            }
        )
        self._store.save(failed)
        return failed


def _branch_name(execution_spec: AgentExecutionSpec, workspace_id: WorkspaceId) -> str:
    return (
        f"evoweave_ds/{execution_spec.run_id}/{execution_spec.task_id}/"
        f"{str(workspace_id).removeprefix('workspace_')[:16]}"
    )


def _validate_binding(lease: WorkspaceLease, spec: AgentExecutionSpec) -> None:
    values_match = (
        lease.run_id == spec.run_id
        and lease.task_id == spec.task_id
        and lease.agent_id == spec.agent_id
        and lease.execution_spec_id == spec.spec_id
        and lease.execution_spec_version == spec.version
        and lease.base_commit == spec.base_commit
        and lease.read_scope == spec.read_scope
        and lease.write_scope == spec.write_scope
    )
    if not values_match:
        raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "工作区租约与执行规格不匹配")
