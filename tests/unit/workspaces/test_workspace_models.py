from pathlib import Path

import pytest
from pydantic import ValidationError

from evoweave_ds.domain.enums import WorkspaceAccessMode, WorkspaceLeaseStatus
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId, WorkspaceId
from evoweave_ds.domain.ports import CommandResult
from evoweave_ds.domain.workspace_models import SandboxPolicy, WorkspaceLease
from evoweave_ds.workspaces.docker_workspace import (
    DockerSandboxConfig,
    DockerWorkspaceCommandRunner,
)


def test_read_only_lease_cannot_have_write_scope(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="只读租约"):
        _lease(tmp_path, write_scope=("src",), access_mode=WorkspaceAccessMode.READ_ONLY)


def test_first_sandbox_policy_cannot_enable_network() -> None:
    with pytest.raises(ValidationError, match="禁止开放网络"):
        SandboxPolicy(network_enabled=True)


def test_command_result_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration_ms"):
        CommandResult(argv=("python",), exit_code=0, duration_ms=-1)


def test_docker_command_is_no_network_and_read_only_for_read_lease(tmp_path: Path) -> None:
    root = tmp_path / "worktree"
    root.mkdir()
    lease = _lease(root)
    runner = DockerWorkspaceCommandRunner(
        lease=lease,
        allowed_commands=("python",),
        config=DockerSandboxConfig(image="python:3.12-slim"),
    )

    argv = runner.build_argv(("python", "-m", "pytest"))

    assert "--network=none" in argv
    assert "--pull=never" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert any(item.endswith(",readonly") for item in argv)
    assert argv[-3:] == ("python", "-m", "pytest")


def _lease(
    root: Path,
    *,
    write_scope: tuple[str, ...] = (),
    access_mode: WorkspaceAccessMode = WorkspaceAccessMode.READ_ONLY,
) -> WorkspaceLease:
    return WorkspaceLease(
        workspace_id=WorkspaceId.new(),
        run_id=RunId.new(),
        task_id=TaskId.new(),
        agent_id=AgentId.new(),
        execution_spec_id=SpecId.new(),
        execution_spec_version=1,
        repository_root=str(root),
        worktree_path=str(root),
        branch_name="evoweave_ds/test/lease",
        base_commit="a" * 40,
        access_mode=access_mode,
        read_scope=("src",),
        write_scope=write_scope,
        status=WorkspaceLeaseStatus.ACTIVE,
    )
