from collections.abc import Callable
from pathlib import Path

import pytest

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import ArtifactKind, ModelAvailability, WorkspaceLeaseStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelRoutingDecision
from evoweave_ds.infrastructure.artifacts.memory import InMemoryArtifactStore
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.workspaces.manager import WorkspaceManager
from evoweave_ds.workspaces.patch_collector import GitPatchCollector
from evoweave_ds.workspaces.state_store import JsonWorkspaceLeaseStore


def test_two_workers_are_isolated_from_each_other_and_main_worktree(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    base_commit = GitInspector(repository).base_commit
    manager = _manager(repository, tmp_path)
    first_spec = _spec(base_commit, write_scope=("calculator.py",))
    second_spec = _spec(base_commit, write_scope=("calculator.py",))
    first_lease = manager.create(first_spec)
    second_lease = manager.create(second_spec)
    try:
        first = manager.open(first_lease, first_spec)
        second = manager.open(second_lease, second_spec)
        first.write_text("calculator.py", "VALUE = 'worker-a'\n")

        assert "worker-a" in first.read_text("calculator.py")
        assert "worker-a" not in second.read_text("calculator.py")
        assert "worker-a" not in (repository / "calculator.py").read_text(encoding="utf-8")
        assert first_lease.branch_name != second_lease.branch_name
        assert first_lease.worktree_path != second_lease.worktree_path
    finally:
        manager.release(first_lease.workspace_id)
        manager.release(second_lease.workspace_id)


def test_patch_is_bound_to_commit_task_agent_spec_and_exact_paths(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    base_commit = GitInspector(repository).base_commit
    manager = _manager(repository, tmp_path)
    spec = _spec(base_commit, write_scope=("calculator.py",))
    lease = manager.create(spec)
    try:
        workspace = manager.open(lease, spec)
        workspace.write_text(
            "calculator.py",
            workspace.read_text("calculator.py").replace("0.9", "0.8"),
        )
        store = InMemoryArtifactStore()
        command_log = store.put_bytes(
            b"pytest passed",
            media_type="text/plain",
            kind=ArtifactKind.COMMAND_LOG,
        )
        patch = GitPatchCollector(store).collect(
            lease=lease,
            execution_spec=spec,
            supporting_artifacts=(command_log,),
        )
        content = store.get_bytes(patch.ref.artifact_id).decode("utf-8")

        assert patch.base_commit == base_commit
        assert patch.task_id == spec.task_id
        assert patch.agent_id == spec.agent_id
        assert patch.execution_spec_id == spec.spec_id
        assert patch.execution_spec_version == spec.version
        assert patch.workspace_id == lease.workspace_id
        assert patch.changed_paths == ("calculator.py",)
        assert patch.supporting_artifact_ids == (command_log.artifact_id,)
        assert "-        return total * 0.9" in content
        assert "+        return total * 0.8" in content
    finally:
        manager.release(lease.workspace_id)


def test_new_untracked_file_is_included_in_patch(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    manager = _manager(repository, tmp_path)
    spec = _spec(
        GitInspector(repository).base_commit,
        write_scope=("generated",),
        read_scope=("calculator.py", "tests", "generated"),
    )
    lease = manager.create(spec)
    try:
        workspace = manager.open(lease, spec)
        workspace.write_text("generated/result.py", "ANSWER = 42\n")
        store = InMemoryArtifactStore()
        patch = GitPatchCollector(store).collect(lease=lease, execution_spec=spec)
        content = store.get_bytes(patch.ref.artifact_id).decode("utf-8")

        assert patch.changed_paths == ("generated/result.py",)
        assert "new file mode" in content
        assert "+ANSWER = 42" in content
    finally:
        manager.release(lease.workspace_id)


def test_workspace_rejects_traversal_absolute_path_and_out_of_scope_write(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    manager = _manager(repository, tmp_path)
    spec = _spec(GitInspector(repository).base_commit, write_scope=("calculator.py",))
    lease = manager.create(spec)
    try:
        workspace = manager.open(lease, spec)
        for path in ("../outside.py", str((tmp_path / "outside.py").resolve())):
            with pytest.raises(DomainError) as error:
                workspace.read_text(path)
            assert error.value.code is ErrorCode.WORKSPACE_ACCESS_DENIED
        with pytest.raises(DomainError) as error:
            workspace.write_text("tests/test_calculator.py", "changed")
        assert error.value.code is ErrorCode.WORKSPACE_ACCESS_DENIED
    finally:
        manager.release(lease.workspace_id)


def test_read_only_workspace_rejects_all_writes(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    manager = _manager(repository, tmp_path)
    spec = _spec(GitInspector(repository).base_commit, write_scope=())
    lease = manager.create(spec)
    try:
        workspace = manager.open(lease, spec)
        with pytest.raises(DomainError) as error:
            workspace.write_text("calculator.py", "forbidden")
        assert error.value.code is ErrorCode.WORKSPACE_ACCESS_DENIED
    finally:
        manager.release(lease.workspace_id)


def test_patch_collector_rejects_direct_change_outside_write_scope(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    manager = _manager(repository, tmp_path)
    spec = _spec(GitInspector(repository).base_commit, write_scope=("calculator.py",))
    lease = manager.create(spec)
    try:
        target = Path(lease.worktree_path) / "tests" / "test_calculator.py"
        target.write_text("direct bypass attempt\n", encoding="utf-8")
        with pytest.raises(DomainError) as error:
            GitPatchCollector(InMemoryArtifactStore()).collect(
                lease=lease,
                execution_spec=spec,
            )
        assert error.value.code is ErrorCode.PATCH_REJECTED
    finally:
        manager.release(lease.workspace_id)


def test_release_is_idempotent_and_persists_recoverable_state(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    manager = _manager(repository, tmp_path)
    spec = _spec(GitInspector(repository).base_commit, write_scope=("calculator.py",))
    lease = manager.create(spec)
    worktree_path = Path(lease.worktree_path)

    released = manager.release(lease.workspace_id)
    released_again = manager.release(lease.workspace_id)

    assert released.status is WorkspaceLeaseStatus.RELEASED
    assert released_again == released
    assert not worktree_path.exists()


def test_session_releases_worktree_when_worker_raises(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    manager = _manager(repository, tmp_path)
    spec = _spec(GitInspector(repository).base_commit, write_scope=("calculator.py",))
    captured_leases = []

    def fail_worker() -> None:
        with manager.session(spec) as (lease, workspace):
            captured_leases.append(lease)
            workspace.write_text("calculator.py", "temporary\n")
            raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        fail_worker()

    captured_lease = captured_leases[0]
    persisted = JsonWorkspaceLeaseStore(tmp_path / "state").get(captured_lease.workspace_id)
    assert persisted.status is WorkspaceLeaseStatus.RELEASED
    assert not Path(captured_lease.worktree_path).exists()


def test_active_lease_can_be_recovered_by_new_manager_instance(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    worktree_root = tmp_path / "worktrees"
    state_root = tmp_path / "state"
    store = JsonWorkspaceLeaseStore(state_root)
    manager = WorkspaceManager(
        repository_root=repository,
        worktree_root=worktree_root,
        lease_store=store,
    )
    spec = _spec(GitInspector(repository).base_commit, write_scope=("calculator.py",))
    lease = manager.create(spec)
    restarted = WorkspaceManager(
        repository_root=repository,
        worktree_root=worktree_root,
        lease_store=JsonWorkspaceLeaseStore(state_root),
    )
    try:
        recovered = restarted.recover()
        workspace = restarted.for_execution(spec)
        assert lease in recovered
        assert workspace.read_text("calculator.py")
    finally:
        restarted.release(lease.workspace_id)


def _manager(repository: Path, tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(
        repository_root=repository,
        worktree_root=tmp_path / "worktrees",
        lease_store=JsonWorkspaceLeaseStore(tmp_path / "state"),
    )


def _spec(
    base_commit: str,
    *,
    write_scope: tuple[str, ...],
    read_scope: tuple[str, ...] = ("calculator.py", "tests"),
) -> AgentExecutionSpec:
    task_id = TaskId.new()
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=task_id,
        task_spec_id=SpecId.new(),
        task_spec_version=1,
        base_commit=base_commit,
        goal="在隔离 worktree 中更新折扣逻辑",
        acceptance_criteria=("补丁准确",),
        model_routing=ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=SpecId.new(),
            requirement_version=1,
            selected_model_key="fake:text",
            selected_availability=ModelAvailability.AVAILABLE,
            reason="测试",
        ),
        tool_names=("file.read", "file.write"),
        allowed_commands=("python",),
        read_scope=read_scope,
        write_scope=write_scope,
    )
