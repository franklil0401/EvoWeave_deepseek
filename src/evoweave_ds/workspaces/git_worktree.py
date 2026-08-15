"""Low-level, no-shell Git worktree lifecycle operations."""

import subprocess
from pathlib import Path

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import IntegrationId, WorkspaceId


class GitWorktreeController:
    def __init__(self, repository_root: Path | str, worktree_root: Path | str) -> None:
        self._repository_root = Path(repository_root).resolve()
        self._worktree_root = Path(worktree_root).resolve()
        self._worktree_root.mkdir(parents=True, exist_ok=True)
        common = Path(
            self._git_text("rev-parse", "--path-format=absolute", "--git-common-dir").strip()
        ).resolve()
        if not common.exists():
            raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "无法定位 Git common dir")

    @property
    def repository_root(self) -> Path:
        return self._repository_root

    @property
    def worktree_root(self) -> Path:
        return self._worktree_root

    def path_for(self, workspace_id: WorkspaceId | IntegrationId) -> Path:
        path = (self._worktree_root / str(workspace_id)).resolve()
        if path.parent != self._worktree_root:
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "工作区目标路径越界")
        return path

    def create(
        self,
        *,
        workspace_id: WorkspaceId | IntegrationId,
        branch_name: str,
        base_commit: str,
    ) -> Path:
        _validate_temporary_branch(branch_name)
        path = self.path_for(workspace_id)
        if path.exists():
            raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "工作区目标已经存在")
        self._git("worktree", "add", "-b", branch_name, str(path), base_commit)
        head = self._git_text("-C", str(path), "rev-parse", "HEAD").strip()
        if head != base_commit:
            self.remove(path=path, branch_name=branch_name)
            raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "worktree HEAD 与基线不一致")
        return path

    def is_registered(self, path: Path | str) -> bool:
        expected = Path(path).resolve()
        output = self._git_text("worktree", "list", "--porcelain", "-z")
        records = output.split("\0")
        return any(
            record.startswith("worktree ")
            and Path(record.removeprefix("worktree ")).resolve() == expected
            for record in records
        )

    def remove(self, *, path: Path | str, branch_name: str) -> None:
        _validate_temporary_branch(branch_name)
        resolved = Path(path).resolve()
        if resolved.parent != self._worktree_root:
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "拒绝回收运行根目录之外的路径")
        if self.is_registered(resolved):
            current_branch = self._git_text(
                "-C", str(resolved), "symbolic-ref", "--quiet", "--short", "HEAD"
            ).strip()
            if current_branch != branch_name:
                raise DomainError(
                    ErrorCode.WORKTREE_OPERATION_FAILED,
                    "worktree 当前分支与临时租约不一致，拒绝回收",
                )
            self._git("worktree", "remove", "--force", str(resolved))
        elif resolved.exists():
            raise DomainError(
                ErrorCode.WORKTREE_OPERATION_FAILED,
                "目标目录存在但不属于当前 Git worktree，拒绝删除",
            )
        self._git("worktree", "prune")
        branch_check = self._git(
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{branch_name}",
            check=False,
        )
        if branch_check.returncode == 0:
            self._git("branch", "-D", branch_name)

    def head(self, path: Path | str) -> str:
        return self._git_text("-C", str(Path(path).resolve()), "rev-parse", "HEAD").strip()

    def _git_text(self, *args: str) -> str:
        return self._git(*args).stdout.decode("utf-8", errors="replace")

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        try:
            completed = subprocess.run(
                ("git", "-C", str(self._repository_root), *args),
                check=False,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "Git worktree 操作失败") from exc
        if check and completed.returncode != 0:
            raise DomainError(
                ErrorCode.WORKTREE_OPERATION_FAILED,
                "Git worktree 命令返回非零状态",
                details={
                    "operation": args[0] if args else "unknown",
                    "stderr": completed.stderr.decode("utf-8", errors="replace")[:2_000],
                },
            )
        return completed


def _validate_temporary_branch(branch_name: str) -> None:
    if (
        not branch_name.startswith("evoweave_ds/")
        or ".." in branch_name
        or "//" in branch_name
        or branch_name.endswith(("/", "."))
    ):
        raise DomainError(
            ErrorCode.WORKTREE_OPERATION_FAILED,
            "只能创建或删除 EvoWeave 临时分支",
        )
