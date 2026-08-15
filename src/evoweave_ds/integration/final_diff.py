"""Export the complete integrated diff as a traceable patch artifact."""

import subprocess
from pathlib import Path

from evoweave_ds.domain.enums import ArtifactKind, IntegrationStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.integration_models import IntegratedPatchSet, IntegrationWorkspaceState
from evoweave_ds.domain.ports import ArtifactStore
from evoweave_ds.domain.validation import validate_repository_path


class FinalDiffExporter:
    def __init__(self, artifact_store: ArtifactStore, worktree_root: Path | str) -> None:
        self._artifact_store = artifact_store
        self._worktree_root = Path(worktree_root).resolve(strict=True)

    def export(self, state: IntegrationWorkspaceState) -> IntegratedPatchSet:
        if state.status is not IntegrationStatus.ACTIVE or not state.applied_patches:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "没有可导出的集成补丁")
        root = Path(state.worktree_path).resolve(strict=True)
        if root.parent != self._worktree_root:
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "最终 diff 工作区路径越界")
        if _git_text(root, "rev-parse", "HEAD").strip() != state.head_commit:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "最终 diff HEAD 漂移")
        content = _git(
            root,
            "diff",
            "--binary",
            "--full-index",
            state.base_commit,
            state.head_commit,
        )
        if not content.strip():
            raise DomainError(ErrorCode.PATCH_EMPTY, "最终集成 diff 为空")
        changed_output = _git(
            root,
            "diff",
            "--name-only",
            "-z",
            state.base_commit,
            state.head_commit,
        )
        try:
            changed_paths = tuple(
                sorted(
                    validate_repository_path(value.decode("utf-8", errors="strict"))
                    for value in changed_output.split(b"\0")
                    if value
                )
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise DomainError(ErrorCode.PATCH_REJECTED, "最终 diff 包含不安全路径") from exc
        reference = self._artifact_store.put_bytes(
            content,
            media_type="text/x-diff",
            kind=ArtifactKind.PATCH,
        )
        return IntegratedPatchSet(
            integration_id=state.integration_id,
            run_id=state.run_id,
            base_commit=state.base_commit,
            candidate_commit=state.head_commit,
            ref=reference,
            changed_paths=changed_paths,
            source_artifact_ids=tuple(item.artifact_id for item in state.applied_patches),
        )


def _git_text(root: Path, *args: str) -> str:
    return _git(root, *args).decode("utf-8", errors="replace")


def _git(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args),
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "导出最终 diff 失败") from exc
    if completed.returncode != 0:
        raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "Git 最终 diff 命令失败")
    return completed.stdout
