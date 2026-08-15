"""Docker command adapter with a read-only container root and default no network."""

from pathlib import Path

from pydantic import Field

from evoweave_ds.capabilities.command_policy import CommandPolicy
from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import WorkspaceLeaseStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.ports import CommandResult
from evoweave_ds.domain.workspace_models import SandboxPolicy, WorkspaceLease
from evoweave_ds.workspaces.command_policy import run_bounded_process, sanitized_environment


class DockerSandboxConfig(DomainModel):
    image: str = Field(
        min_length=1,
        max_length=512,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/:@-]{0,511}$",
    )
    docker_executable: str = Field(default="docker", pattern=r"^[A-Za-z0-9_.-]+$")


class DockerWorkspaceCommandRunner:
    def __init__(
        self,
        *,
        lease: WorkspaceLease,
        allowed_commands: tuple[str, ...],
        config: DockerSandboxConfig,
        policy: SandboxPolicy | None = None,
    ) -> None:
        if lease.status is not WorkspaceLeaseStatus.ACTIVE:
            raise DomainError(ErrorCode.WORKSPACE_STATE_INVALID, "命令执行需要 active 租约")
        self._lease = lease
        self._root = Path(lease.worktree_path).resolve(strict=True)
        self._allowed_commands = allowed_commands
        self._config = config
        self._policy = policy or SandboxPolicy()
        self._command_policy = CommandPolicy()
        if "," in str(self._root):
            raise DomainError(ErrorCode.SANDBOX_UNAVAILABLE, "Docker 挂载路径不能包含逗号")

    def build_argv(self, argv: tuple[str, ...]) -> tuple[str, ...]:
        self._command_policy.authorize(argv, allowed_commands=self._allowed_commands)
        mount = f"type=bind,source={self._root},target=/workspace"
        if not self._lease.write_scope:
            mount += ",readonly"
        return (
            self._config.docker_executable,
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--cpus={self._policy.max_cpus}",
            f"--memory={self._policy.max_memory_mb}m",
            f"--pids-limit={self._policy.max_processes}",
            "--mount",
            mount,
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--workdir=/workspace",
            "--env=CI=1",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            self._config.image,
            *argv,
        )

    def run(self, argv: tuple[str, ...], *, timeout_seconds: int) -> CommandResult:
        if timeout_seconds > self._policy.max_timeout_seconds:
            raise DomainError(ErrorCode.COMMAND_DENIED, "命令超时参数超过沙箱策略上限")
        docker_argv = self.build_argv(argv)
        result = run_bounded_process(
            docker_argv,
            cwd=self._root,
            timeout_seconds=timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
            environment=sanitized_environment(),
        )
        return CommandResult(
            argv=argv,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
            output_truncated=result.output_truncated,
        )
