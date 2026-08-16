"""Trusted-local command adapter with sanitized environment and bounded output."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import monotonic
from typing import Protocol

from evoweave_ds.capabilities.command_policy import CommandPolicy
from evoweave_ds.domain.enums import WorkspaceLeaseStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.ports import CommandResult
from evoweave_ds.domain.workspace_models import SandboxPolicy, WorkspaceLease

_SAFE_ENVIRONMENT_NAMES = ("PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "TMP", "WINDIR")


class _ReadableTemporaryFile(Protocol):
    def seek(self, offset: int, whence: int = 0) -> int: ...

    def tell(self) -> int: ...

    def read(self, size: int = -1) -> bytes: ...


class LocalWorkspaceCommandRunner:
    """Host executor for trusted fixtures only; production should use a sandbox adapter."""

    def __init__(
        self,
        *,
        lease: WorkspaceLease,
        allowed_commands: tuple[str, ...],
        policy: SandboxPolicy | None = None,
        allow_host_execution: bool = False,
    ) -> None:
        if lease.status is not WorkspaceLeaseStatus.ACTIVE:
            raise DomainError(ErrorCode.WORKSPACE_STATE_INVALID, "命令执行需要 active 租约")
        if not allow_host_execution:
            raise DomainError(
                ErrorCode.SANDBOX_UNAVAILABLE,
                "本地命令执行默认关闭；请显式使用沙箱适配器",
            )
        self._root = Path(lease.worktree_path).resolve(strict=True)
        self._allowed_commands = allowed_commands
        self._policy = policy or SandboxPolicy()
        self._command_policy = CommandPolicy()

    def run(self, argv: tuple[str, ...], *, timeout_seconds: int) -> CommandResult:
        self._command_policy.authorize(argv, allowed_commands=self._allowed_commands)
        if timeout_seconds > self._policy.max_timeout_seconds:
            raise DomainError(ErrorCode.COMMAND_DENIED, "命令超时参数超过沙箱策略上限")
        environment = sanitized_environment()
        # src 布局仓库(如 Flask)的测试需要 PYTHONPATH=src 才能 import 项目包;
        # web_learning_tool 这类根布局仓库无 src 目录, 该注入无影响。
        src_dir = self._root / "src"
        if src_dir.is_dir():
            existing = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = (
                str(src_dir) + (";" + existing if existing else "")
            )
        executable = shutil.which(argv[0], path=environment.get("PATH"))
        if executable is None:
            raise DomainError(ErrorCode.COMMAND_DENIED, "找不到授权命令的可执行文件")
        result = run_bounded_process(
            (executable, *argv[1:]),
            cwd=self._root,
            timeout_seconds=timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
            environment=environment,
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


def run_bounded_process(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: int,
    max_output_bytes: int,
    environment: dict[str, str],
) -> CommandResult:
    start = monotonic()
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                shell=False,
                creationflags=creation_flags,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise DomainError(
                ErrorCode.COMMAND_DENIED,
                "无法启动授权命令",
                details={"executable": argv[0]},
            ) from exc
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)
            exit_code = -1
        duration_ms = int((monotonic() - start) * 1_000)
        stdout, stdout_truncated = _read_bounded(stdout_file, max_output_bytes // 2)
        stderr, stderr_truncated = _read_bounded(stderr_file, max_output_bytes // 2)
    return CommandResult(
        argv=argv,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_ms=duration_ms,
        output_truncated=stdout_truncated or stderr_truncated,
    )


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ("taskkill", "/PID", str(process.pid), "/T", "/F"),
            check=False,
            capture_output=True,
            timeout=10,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _read_bounded(stream: _ReadableTemporaryFile, limit: int) -> tuple[str, bool]:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    data = stream.read(limit)
    return data.decode("utf-8", errors="replace"), size > limit


def sanitized_environment() -> dict[str, str]:
    environment = {name: os.environ[name] for name in _SAFE_ENVIRONMENT_NAMES if name in os.environ}
    environment.update(
        {
            "CI": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return environment
