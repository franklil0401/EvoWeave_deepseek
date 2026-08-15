import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import (
    InputModality,
    IntegrationStatus,
    ModelAvailability,
    TaskDifficulty,
)
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import (
    DifficultyAssessment,
    ModelRequirement,
    ModelRoutingDecision,
)
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.infrastructure.artifacts.memory import InMemoryArtifactStore
from evoweave_ds.integration.final_diff import FinalDiffExporter
from evoweave_ds.integration.integration_workspace import IntegrationWorkspaceManager
from evoweave_ds.integration.patch_applier import PatchApplier
from evoweave_ds.integration.patch_guard import PatchGuard
from evoweave_ds.integration.rollback import IntegrationRollback
from evoweave_ds.integration.service import PatchIntegrationService
from evoweave_ds.integration.state_store import JsonIntegrationStateStore
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.workspaces.manager import WorkspaceManager
from evoweave_ds.workspaces.patch_collector import GitPatchCollector
from evoweave_ds.workspaces.state_store import JsonWorkspaceLeaseStore


def test_disjoint_patches_follow_dependency_order_and_latest_patch_rolls_back(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    base_commit = GitInspector(repository).base_commit
    artifact_store = InMemoryArtifactStore()
    change_spec_id = SpecId.new()
    first = _task(
        base_commit,
        change_spec_id=change_spec_id,
        write_scope=("calculator.py",),
    )
    second = _task(
        base_commit,
        change_spec_id=change_spec_id,
        write_scope=("tests/test_calculator.py",),
        depends_on=(first.task_id,),
    )
    first_patch = _collect_patch(
        repository,
        tmp_path,
        artifact_store,
        first,
        "calculator.py",
        lambda text: text.replace("0.9", "0.8"),
    )
    second_patch = _collect_patch(
        repository,
        tmp_path,
        artifact_store,
        second,
        "tests/test_calculator.py",
        lambda text: f"# integrated second\n{text}",
    )
    manager, service, applier, _state_store = _integration_components(
        repository,
        tmp_path,
        artifact_store,
    )

    state = service.integrate(
        run_id=RunId.new(),
        base_commit=base_commit,
        patches=(second_patch, first_patch),
        task_specs=(first, second),
    )
    root = Path(state.worktree_path)
    assert [item.task_id for item in state.applied_patches] == [first.task_id, second.task_id]
    assert "0.8" in (root / "calculator.py").read_text(encoding="utf-8")
    assert "integrated second" in (root / "tests/test_calculator.py").read_text(encoding="utf-8")
    assert "0.8" not in (repository / "calculator.py").read_text(encoding="utf-8")
    assert _git(repository, "branch", "--show-current") == "main"
    final_patch = FinalDiffExporter(artifact_store, manager.worktree_root).export(state)
    assert final_patch.changed_paths == ("calculator.py", "tests/test_calculator.py")
    assert final_patch.source_artifact_ids == (
        first_patch.ref.artifact_id,
        second_patch.ref.artifact_id,
    )
    assert artifact_store.get_bytes(final_patch.ref.artifact_id)

    rolled_back = IntegrationRollback(manager, applier).latest(state.integration_id)
    assert len(rolled_back.applied_patches) == 1
    assert "0.8" in (root / "calculator.py").read_text(encoding="utf-8")
    assert "integrated second" not in (root / "tests/test_calculator.py").read_text(
        encoding="utf-8"
    )
    manager.release(state.integration_id)
    assert not root.exists()


def test_overlapping_patch_set_is_rejected_before_any_patch_is_kept(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    base_commit = GitInspector(repository).base_commit
    artifact_store = InMemoryArtifactStore()
    change_spec_id = SpecId.new()
    first = _task(base_commit, change_spec_id=change_spec_id, write_scope=("calculator.py",))
    second = _task(base_commit, change_spec_id=change_spec_id, write_scope=("calculator.py",))
    first_patch = _collect_patch(
        repository,
        tmp_path,
        artifact_store,
        first,
        "calculator.py",
        lambda text: text.replace("0.9", "0.8"),
    )
    second_patch = _collect_patch(
        repository,
        tmp_path,
        artifact_store,
        second,
        "calculator.py",
        lambda text: text.replace("0.9", "0.7"),
    )
    manager, service, _applier, state_store = _integration_components(
        repository,
        tmp_path,
        artifact_store,
    )

    with pytest.raises(DomainError) as error:
        service.integrate(
            run_id=RunId.new(),
            base_commit=base_commit,
            patches=(first_patch, second_patch),
            task_specs=(first, second),
        )

    assert error.value.code is ErrorCode.PATCH_CONFLICT
    state = state_store.list_all()[0]
    assert state.status is IntegrationStatus.FAILED
    assert state.applied_patches == ()
    assert "0.9" in (Path(state.worktree_path) / "calculator.py").read_text(encoding="utf-8")
    manager.release(state.integration_id)


@pytest.mark.parametrize("case", ["scope", "sensitive"])
def test_patch_guard_rejects_out_of_scope_and_sensitive_changes(
    case: str,
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    base_commit = GitInspector(repository).base_commit
    artifact_store = InMemoryArtifactStore()
    change_spec_id = SpecId.new()
    worker_scope = ("calculator.py",) if case == "scope" else (".env",)
    task = _task(base_commit, change_spec_id=change_spec_id, write_scope=worker_scope)
    path = "calculator.py" if case == "scope" else ".env"
    patch = _collect_patch(
        repository,
        tmp_path,
        artifact_store,
        task,
        path,
        (lambda text: text.replace("0.9", "0.8"))
        if case == "scope"
        else (lambda _text: "TOKEN=x\n"),
        create=case == "sensitive",
    )
    integration_task = (
        task.model_copy(update={"write_scope": ("tests",), "read_scope": ("tests",)})
        if case == "scope"
        else task
    )
    manager, service, _applier, state_store = _integration_components(
        repository,
        tmp_path,
        artifact_store,
    )

    with pytest.raises(DomainError) as error:
        service.integrate(
            run_id=RunId.new(),
            base_commit=base_commit,
            patches=(patch,),
            task_specs=(integration_task,),
        )

    assert error.value.code is ErrorCode.PATCH_REJECTED
    state = state_store.list_all()[0]
    assert state.applied_patches == ()
    manager.release(state.integration_id)


def test_patch_base_drift_is_rejected(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    repository = committed_repository("single_module")
    original_base = GitInspector(repository).base_commit
    artifact_store = InMemoryArtifactStore()
    task = _task(original_base, change_spec_id=SpecId.new(), write_scope=("calculator.py",))
    patch = _collect_patch(
        repository,
        tmp_path,
        artifact_store,
        task,
        "calculator.py",
        lambda text: text.replace("0.9", "0.8"),
    )
    (repository / "README.txt").write_text("new base\n", encoding="utf-8")
    _git(repository, "add", "README.txt")
    _git(repository, "commit", "-m", "advance base")
    new_base = GitInspector(repository).base_commit
    replacement_task = task.model_copy(update={"base_commit": new_base})
    manager, service, _applier, state_store = _integration_components(
        repository,
        tmp_path,
        artifact_store,
    )

    with pytest.raises(DomainError) as error:
        service.integrate(
            run_id=RunId.new(),
            base_commit=new_base,
            patches=(patch,),
            task_specs=(replacement_task,),
        )

    assert error.value.code is ErrorCode.PATCH_BASE_MISMATCH
    state = state_store.list_all()[0]
    assert state.applied_patches == ()
    manager.release(state.integration_id)


def _integration_components(
    repository: Path,
    tmp_path: Path,
    artifact_store: InMemoryArtifactStore,
) -> tuple[
    IntegrationWorkspaceManager,
    PatchIntegrationService,
    PatchApplier,
    JsonIntegrationStateStore,
]:
    state_store = JsonIntegrationStateStore(tmp_path / "integration-state")
    manager = IntegrationWorkspaceManager(
        repository_root=repository,
        worktree_root=tmp_path / "integration-worktrees",
        state_store=state_store,
    )
    applier = PatchApplier(manager.worktree_root)
    service = PatchIntegrationService(
        manager=manager,
        guard=PatchGuard(artifact_store),
        applier=applier,
    )
    return manager, service, applier, state_store


def _collect_patch(
    repository: Path,
    tmp_path: Path,
    artifact_store: InMemoryArtifactStore,
    task: TaskSpec,
    path: str,
    transform: Callable[[str], str],
    *,
    create: bool = False,
):
    manager = WorkspaceManager(
        repository_root=repository,
        worktree_root=tmp_path / "worker-worktrees",
        lease_store=JsonWorkspaceLeaseStore(tmp_path / "worker-state"),
    )
    execution = _execution(task)
    lease = manager.create(execution)
    try:
        workspace = manager.open(lease, execution)
        original = "" if create else workspace.read_text(path)
        workspace.write_text(path, transform(original))
        return GitPatchCollector(artifact_store).collect(
            lease=lease,
            execution_spec=execution,
        )
    finally:
        manager.release(lease.workspace_id)


def _task(
    base_commit: str,
    *,
    change_spec_id: SpecId,
    write_scope: tuple[str, ...],
    depends_on: tuple[TaskId, ...] = (),
) -> TaskSpec:
    task_id = TaskId.new()
    return TaskSpec(
        spec_id=SpecId.new(),
        task_id=task_id,
        change_spec_id=change_spec_id,
        goal="生成可集成补丁",
        base_commit=base_commit,
        acceptance_criteria=("补丁通过门禁",),
        depends_on=depends_on,
        read_scope=write_scope,
        write_scope=write_scope,
        difficulty=DifficultyAssessment(difficulty=TaskDifficulty.LOW, rationale="测试"),
        model_requirement=ModelRequirement(
            requirement_id=SpecId.new(),
            task_id=task_id,
            difficulty=TaskDifficulty.LOW,
            required_modalities=(InputModality.TEXT,),
        ),
    )


def _execution(task: TaskSpec) -> AgentExecutionSpec:
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=task.task_id,
        task_spec_id=task.spec_id,
        task_spec_version=task.version,
        base_commit=task.base_commit,
        goal=task.goal,
        acceptance_criteria=task.acceptance_criteria,
        model_routing=ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=task.model_requirement.requirement_id,
            requirement_version=task.model_requirement.version,
            selected_model_key="fake:text",
            selected_availability=ModelAvailability.AVAILABLE,
            reason="测试",
        ),
        tool_names=("file.read", "file.write"),
        read_scope=task.read_scope,
        write_scope=task.write_scope,
    )


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
