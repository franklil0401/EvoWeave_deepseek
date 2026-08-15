"""Lifecycle for a dedicated branch and worktree used only for patch integration."""

from pathlib import Path
from threading import RLock

from evoweave_ds.domain.base import utc_now
from evoweave_ds.domain.enums import IntegrationStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import IntegrationId, RunId
from evoweave_ds.domain.integration_models import IntegrationWorkspaceState
from evoweave_ds.integration.state_store import JsonIntegrationStateStore
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.workspaces.git_worktree import GitWorktreeController


class IntegrationWorkspaceManager:
    def __init__(
        self,
        *,
        repository_root: Path | str,
        worktree_root: Path | str,
        state_store: JsonIntegrationStateStore,
    ) -> None:
        inspector = GitInspector(repository_root)
        self._repository_root = inspector.repository_root
        self._controller = GitWorktreeController(self._repository_root, worktree_root)
        self._store = state_store
        self._lock = RLock()

    @property
    def worktree_root(self) -> Path:
        return self._controller.worktree_root

    def create(self, *, run_id: RunId, base_commit: str) -> IntegrationWorkspaceState:
        with self._lock:
            resolved_base = GitInspector(self._repository_root, base_commit).base_commit
            if resolved_base != base_commit:
                raise DomainError(ErrorCode.PATCH_BASE_MISMATCH, "无法解析集成基线")
            integration_id = IntegrationId.new()
            branch_name = (
                f"evoweave_ds/integration/{run_id}/"
                f"{str(integration_id).removeprefix('integration_')[:16]}"
            )
            path = self._controller.path_for(integration_id)
            state = IntegrationWorkspaceState(
                integration_id=integration_id,
                run_id=run_id,
                repository_root=str(self._repository_root),
                worktree_path=str(path),
                branch_name=branch_name,
                base_commit=base_commit,
                head_commit=base_commit,
            )
            self._store.save(state)
            try:
                self._controller.create(
                    workspace_id=integration_id,
                    branch_name=branch_name,
                    base_commit=base_commit,
                )
            except DomainError as exc:
                failed = state.model_copy(
                    update={
                        "status": IntegrationStatus.FAILED,
                        "failure_reason": exc.message,
                        "updated_at": utc_now(),
                        "version": state.version + 1,
                    }
                )
                self._store.save(failed)
                raise
            active = state.model_copy(
                update={
                    "status": IntegrationStatus.ACTIVE,
                    "updated_at": utc_now(),
                    "version": state.version + 1,
                }
            )
            self._store.save(active)
            return active

    def get(self, integration_id: IntegrationId) -> IntegrationWorkspaceState:
        return self._store.get(integration_id)

    def save(self, state: IntegrationWorkspaceState) -> None:
        if state.status is not IntegrationStatus.ACTIVE:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "只能保存 active 集成进度")
        self._assert_registered_head(state)
        self._store.save(state)

    def mark_failed(
        self,
        integration_id: IntegrationId,
        reason: str,
    ) -> IntegrationWorkspaceState:
        with self._lock:
            state = self._store.get(integration_id)
            if state.status is IntegrationStatus.FAILED:
                return state
            if state.status is not IntegrationStatus.ACTIVE:
                raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "当前集成状态不能标记失败")
            failed = state.model_copy(
                update={
                    "status": IntegrationStatus.FAILED,
                    "failure_reason": reason[:2_000],
                    "updated_at": utc_now(),
                    "version": state.version + 1,
                }
            )
            self._store.save(failed)
            return failed

    def release(self, integration_id: IntegrationId) -> IntegrationWorkspaceState:
        with self._lock:
            state = self._store.get(integration_id)
            if state.status is IntegrationStatus.RELEASED:
                return state
            if state.status not in {
                IntegrationStatus.ACTIVE,
                IntegrationStatus.FAILED,
                IntegrationStatus.RELEASING,
            }:
                raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "当前集成状态不能回收")
            if state.status is IntegrationStatus.RELEASING:
                releasing = state
            else:
                releasing = state.model_copy(
                    update={
                        "status": IntegrationStatus.RELEASING,
                        "failure_reason": None,
                        "updated_at": utc_now(),
                        "version": state.version + 1,
                    }
                )
                self._store.save(releasing)
            self._controller.remove(path=state.worktree_path, branch_name=state.branch_name)
            released = releasing.model_copy(
                update={
                    "status": IntegrationStatus.RELEASED,
                    "updated_at": utc_now(),
                    "version": releasing.version + 1,
                }
            )
            self._store.save(released)
            return released

    def recover(self) -> tuple[IntegrationWorkspaceState, ...]:
        recovered: list[IntegrationWorkspaceState] = []
        with self._lock:
            for state in self._store.list_all():
                registered = self._controller.is_registered(state.worktree_path)
                if state.status is IntegrationStatus.CREATING:
                    if (
                        registered
                        and self._controller.head(state.worktree_path) == state.head_commit
                    ):
                        revised = state.model_copy(
                            update={
                                "status": IntegrationStatus.ACTIVE,
                                "updated_at": utc_now(),
                                "version": state.version + 1,
                            }
                        )
                    else:
                        revised = state.model_copy(
                            update={
                                "status": IntegrationStatus.FAILED,
                                "failure_reason": "集成工作区创建过程被中断",
                                "updated_at": utc_now(),
                                "version": state.version + 1,
                            }
                        )
                    self._store.save(revised)
                    recovered.append(revised)
                elif state.status is IntegrationStatus.ACTIVE:
                    if (
                        registered
                        and self._controller.head(state.worktree_path) == state.head_commit
                    ):
                        recovered.append(state)
                    else:
                        recovered.append(
                            self.mark_failed(state.integration_id, "集成 worktree 丢失或 HEAD 漂移")
                        )
                elif state.status is IntegrationStatus.RELEASING:
                    recovered.append(self.release(state.integration_id))
        return tuple(recovered)

    def _assert_registered_head(self, state: IntegrationWorkspaceState) -> None:
        if not self._controller.is_registered(state.worktree_path):
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "集成 worktree 未注册")
        if self._controller.head(state.worktree_path) != state.head_commit:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "集成 worktree HEAD 漂移")
