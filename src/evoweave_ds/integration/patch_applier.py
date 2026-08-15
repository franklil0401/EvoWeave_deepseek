"""Apply one guarded patch as one commit and support latest-patch rollback."""

import subprocess
from pathlib import Path

from evoweave_ds.domain.base import utc_now
from evoweave_ds.domain.enums import IntegrationStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.integration_models import (
    AppliedPatchRecord,
    GuardedPatch,
    IntegrationWorkspaceState,
)
from evoweave_ds.domain.validation import validate_repository_path
from evoweave_ds.integration.patch_guard import PatchGuard


class PatchApplier:
    def __init__(self, worktree_root: Path | str) -> None:
        self._worktree_root = Path(worktree_root).resolve(strict=True)

    def apply(
        self,
        state: IntegrationWorkspaceState,
        guarded: GuardedPatch,
        content: bytes,
    ) -> IntegrationWorkspaceState:
        root = self._validated_root(state)
        self._assert_clean_head(root, state.head_commit)
        PatchGuard.check_applies(root, content)
        before_commit = state.head_commit
        try:
            applied = _git(
                root,
                "apply",
                "--index",
                "--binary",
                "--whitespace=error-all",
                "-",
                input_bytes=content,
            )
            if applied.returncode != 0:
                raise DomainError(
                    ErrorCode.PATCH_CONFLICT,
                    "补丁应用失败",
                    details={"stderr": _stderr(applied)},
                )
            staged_paths = _staged_paths(root)
            if set(staged_paths) != set(guarded.parsed_paths):
                raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "暂存路径与已守卫补丁不一致")
            committed = _git(
                root,
                "-c",
                "user.name=EvoWeave Integration",
                "-c",
                "user.email=integration@evoweave_ds.local",
                "commit",
                "--no-gpg-sign",
                "-m",
                f"evoweave_ds: integrate {guarded.artifact.task_id}",
            )
            if committed.returncode != 0:
                raise DomainError(
                    ErrorCode.PATCH_CONFLICT,
                    "补丁提交失败",
                    details={"stderr": _stderr(committed)},
                )
            after_commit = _git_text(root, "rev-parse", "HEAD").strip()
            record = AppliedPatchRecord(
                sequence=len(state.applied_patches) + 1,
                artifact_id=guarded.artifact.ref.artifact_id,
                task_id=guarded.artifact.task_id,
                before_commit=before_commit,
                after_commit=after_commit,
                changed_paths=guarded.parsed_paths,
            )
            return state.model_copy(
                update={
                    "head_commit": after_commit,
                    "applied_patches": (*state.applied_patches, record),
                    "updated_at": utc_now(),
                    "version": state.version + 1,
                }
            )
        except Exception:
            self._restore(root, before_commit)
            raise

    def rollback_latest(self, state: IntegrationWorkspaceState) -> IntegrationWorkspaceState:
        if not state.applied_patches:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "没有可回滚的补丁")
        root = self._validated_root(state)
        self._assert_clean_head(root, state.head_commit)
        latest = state.applied_patches[-1]
        self._restore(root, latest.before_commit)
        return state.model_copy(
            update={
                "head_commit": latest.before_commit,
                "applied_patches": state.applied_patches[:-1],
                "updated_at": utc_now(),
                "version": state.version + 1,
            }
        )

    def _validated_root(self, state: IntegrationWorkspaceState) -> Path:
        if state.status is not IntegrationStatus.ACTIVE:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "补丁操作需要 active 集成状态")
        root = Path(state.worktree_path).resolve(strict=True)
        if root.parent != self._worktree_root:
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "集成工作区路径越界")
        return root

    @staticmethod
    def _assert_clean_head(root: Path, expected_head: str) -> None:
        if _git_text(root, "rev-parse", "HEAD").strip() != expected_head:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "集成 HEAD 与状态不一致")
        if _git(root, "status", "--porcelain=v1", "-z").stdout:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "集成工作区存在未记录修改")

    @staticmethod
    def _restore(root: Path, commit: str) -> None:
        reset = _git(root, "reset", "--hard", commit)
        clean = _git(root, "clean", "-fd")
        if reset.returncode != 0 or clean.returncode != 0:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "补丁回滚失败")


def _staged_paths(root: Path) -> tuple[str, ...]:
    output = _git(root, "diff", "--cached", "--name-only", "-z").stdout
    paths: list[str] = []
    for value in output.split(b"\0"):
        if not value:
            continue
        try:
            paths.append(validate_repository_path(value.decode("utf-8", errors="strict")))
        except (UnicodeDecodeError, ValueError) as exc:
            raise DomainError(ErrorCode.PATCH_REJECTED, "暂存区包含不安全路径") from exc
    return tuple(sorted(paths))


def _git_text(root: Path, *args: str) -> str:
    completed = _git(root, *args)
    if completed.returncode != 0:
        raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "Git 集成查询失败")
    return completed.stdout.decode("utf-8", errors="replace")


def _git(
    root: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ("git", "-C", str(root), *args),
            input=input_bytes,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "Git 集成操作失败") from exc


def _stderr(completed: subprocess.CompletedProcess[bytes]) -> str:
    return completed.stderr.decode("utf-8", errors="replace")[:2_000]
