from collections.abc import Callable
from pathlib import Path

import pytest

from evoweave_ds.domain.enums import WorkspaceAccessMode, WorkspaceLeaseStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId, WorkspaceId
from evoweave_ds.domain.workspace_models import SandboxPolicy, WorkspaceLease
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.workspaces.command_policy import LocalWorkspaceCommandRunner


def test_host_command_execution_is_off_by_default(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("single_module")
    lease = _lease(repository, GitInspector(repository).base_commit)
    with pytest.raises(DomainError) as error:
        LocalWorkspaceCommandRunner(lease=lease, allowed_commands=("python",))
    assert error.value.code is ErrorCode.SANDBOX_UNAVAILABLE


def test_trusted_command_records_success_nonzero_timeout_and_truncation(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("single_module")
    lease = _lease(repository, GitInspector(repository).base_commit)
    runner = LocalWorkspaceCommandRunner(
        lease=lease,
        allowed_commands=("python",),
        policy=SandboxPolicy(max_output_bytes=20, max_timeout_seconds=5),
        allow_host_execution=True,
    )

    success = runner.run(("python", "-c", "print('ok')"), timeout_seconds=2)
    nonzero = runner.run(("python", "-c", "raise SystemExit(7)"), timeout_seconds=2)
    timeout = runner.run(
        ("python", "-c", "__import__('time').sleep(2)"),
        timeout_seconds=1,
    )
    truncated = runner.run(("python", "-c", "print('x' * 100)"), timeout_seconds=2)

    assert success.exit_code == 0
    assert success.stdout.strip() == "ok"
    assert nonzero.exit_code == 7
    assert timeout.timed_out is True
    assert timeout.exit_code == -1
    assert truncated.output_truncated is True
    assert len(truncated.stdout.encode()) <= 10


def test_trusted_command_does_not_inherit_model_api_keys(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("single_module")
    lease = _lease(repository, GitInspector(repository).base_commit)
    runner = LocalWorkspaceCommandRunner(
        lease=lease,
        allowed_commands=("python",),
        allow_host_execution=True,
    )
    result = runner.run(
        (
            "python",
            "-c",
            "print(__import__('os').getenv('Deepseek_api_key', 'missing'))",
        ),
        timeout_seconds=2,
    )

    assert result.stdout.strip() == "missing"


def test_command_policy_rejects_unlisted_executable_and_shell_control(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("single_module")
    runner = LocalWorkspaceCommandRunner(
        lease=_lease(repository, GitInspector(repository).base_commit),
        allowed_commands=("python",),
        allow_host_execution=True,
    )
    for argv in (("git", "status"), ("python", "-c", "print(1); rm -rf x")):
        with pytest.raises(DomainError) as error:
            runner.run(argv, timeout_seconds=2)
        assert error.value.code is ErrorCode.COMMAND_DENIED


def _lease(root: Path, base_commit: str) -> WorkspaceLease:
    return WorkspaceLease(
        workspace_id=WorkspaceId.new(),
        run_id=RunId.new(),
        task_id=TaskId.new(),
        agent_id=AgentId.new(),
        execution_spec_id=SpecId.new(),
        execution_spec_version=1,
        repository_root=str(root),
        worktree_path=str(root),
        branch_name="evoweave_ds/test/command",
        base_commit=base_commit,
        access_mode=WorkspaceAccessMode.READ_ONLY,
        read_scope=("calculator.py",),
        status=WorkspaceLeaseStatus.ACTIVE,
    )
