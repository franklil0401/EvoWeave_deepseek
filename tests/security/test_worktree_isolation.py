import os
from collections.abc import Callable
from pathlib import Path

import pytest

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import ModelAvailability
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelRoutingDecision
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.workspaces.manager import WorkspaceManager
from evoweave_ds.workspaces.state_store import JsonWorkspaceLeaseStore


def test_symbolic_link_escape_is_rejected(
    committed_repository: Callable[[str], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = committed_repository("single_module")
    base_commit = GitInspector(repository).base_commit
    manager = WorkspaceManager(
        repository_root=repository,
        worktree_root=tmp_path / "worktrees",
        lease_store=JsonWorkspaceLeaseStore(tmp_path / "state"),
    )
    spec = _spec(base_commit)
    lease = manager.create(spec)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("outside", encoding="utf-8")
    link = Path(lease.worktree_path) / "linked-secret.txt"
    try:
        try:
            os.symlink(outside, link)
        except OSError:
            link.write_text("simulated link", encoding="utf-8")
            monkeypatch.setattr(
                "evoweave_ds.workspaces.path_policy._is_link_or_junction",
                lambda path: path == link,
            )
        workspace = manager.open(lease, spec)
        with pytest.raises(DomainError) as error:
            workspace.read_text("linked-secret.txt")
        assert error.value.code is ErrorCode.WORKSPACE_ACCESS_DENIED
        with pytest.raises(DomainError) as error:
            workspace.write_text("linked-secret.txt", "overwrite")
        assert error.value.code is ErrorCode.WORKSPACE_ACCESS_DENIED
        assert outside.read_text(encoding="utf-8") == "outside"
    finally:
        manager.release(lease.workspace_id)


def _spec(base_commit: str) -> AgentExecutionSpec:
    task_id = TaskId.new()
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=task_id,
        task_spec_id=SpecId.new(),
        task_spec_version=1,
        base_commit=base_commit,
        goal="验证符号链接隔离",
        acceptance_criteria=("不能读取工作区外文件",),
        model_routing=ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=SpecId.new(),
            requirement_version=1,
            selected_model_key="fake:text",
            selected_availability=ModelAvailability.AVAILABLE,
            reason="测试",
        ),
        read_scope=("calculator.py", "linked-secret.txt"),
        write_scope=("linked-secret.txt",),
    )
