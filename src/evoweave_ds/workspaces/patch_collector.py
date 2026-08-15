"""Validate a worktree diff and persist it as a traceable PatchArtifact."""

import subprocess
from pathlib import Path

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.artifacts import ArtifactRef, PatchArtifact
from evoweave_ds.domain.enums import ArtifactKind, WorkspaceLeaseStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.ports import ArtifactStore
from evoweave_ds.domain.validation import path_is_within_scopes, validate_repository_path
from evoweave_ds.domain.workspace_models import WorkspaceLease
from evoweave_ds.workspaces.path_policy import WorkspacePathPolicy


class GitPatchCollector:
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    def collect(
        self,
        *,
        lease: WorkspaceLease,
        execution_spec: AgentExecutionSpec,
        supporting_artifacts: tuple[ArtifactRef, ...] = (),
    ) -> PatchArtifact:
        if lease.status is not WorkspaceLeaseStatus.ACTIVE:
            raise DomainError(ErrorCode.WORKSPACE_STATE_INVALID, "只能从 active 工作区收集补丁")
        _assert_binding(lease, execution_spec)
        root = Path(lease.worktree_path).resolve(strict=True)
        head = _git(root, "rev-parse", "HEAD").stdout.decode().strip()
        if head != lease.base_commit:
            raise DomainError(ErrorCode.PATCH_REJECTED, "工作区 HEAD 已偏离 base commit")
        changed_paths, untracked_paths = _status_paths(root)
        if not changed_paths:
            raise DomainError(ErrorCode.PATCH_EMPTY, "工作区没有可收集的修改")
        policy = WorkspacePathPolicy(
            root=root,
            read_scope=lease.read_scope,
            write_scope=lease.write_scope,
        )
        for path in changed_paths:
            if not path_is_within_scopes(path, lease.write_scope):
                raise DomainError(
                    ErrorCode.PATCH_REJECTED,
                    f"修改文件超出 write_scope：{path}",
                )
            policy.assert_writable_path(path)

        try:
            if untracked_paths:
                _git(root, "add", "--intent-to-add", "--", *untracked_paths)
            patch_bytes = _git(
                root,
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                lease.base_commit,
                "--",
                ".",
            ).stdout
        finally:
            _git(root, "reset", "--mixed", "--quiet", lease.base_commit, "--", ".")
        if not patch_bytes.strip():
            raise DomainError(ErrorCode.PATCH_EMPTY, "Git 未生成非空补丁")
        unsupported = [
            artifact.kind
            for artifact in supporting_artifacts
            if artifact.kind not in {ArtifactKind.COMMAND_LOG, ArtifactKind.TEST_REPORT}
        ]
        if unsupported:
            raise DomainError(
                ErrorCode.PATCH_REJECTED,
                "补丁只允许关联命令日志和测试报告",
                details={"kinds": [item.value for item in unsupported]},
            )
        reference = self._artifact_store.put_bytes(
            patch_bytes,
            media_type="text/x-diff",
            kind=ArtifactKind.PATCH,
        )
        return PatchArtifact(
            ref=reference,
            task_id=execution_spec.task_id,
            agent_id=execution_spec.agent_id,
            execution_spec_id=execution_spec.spec_id,
            execution_spec_version=execution_spec.version,
            workspace_id=lease.workspace_id,
            base_commit=lease.base_commit,
            changed_paths=changed_paths,
            supporting_artifact_ids=tuple(
                artifact.artifact_id for artifact in supporting_artifacts
            ),
        )


def _status_paths(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ).stdout
    records = [record for record in raw.split(b"\0") if record]
    changed: set[str] = set()
    untracked: set[str] = set()
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 4:
            raise DomainError(ErrorCode.PATCH_REJECTED, "无法解析 Git 状态")
        status = record[:2]
        path = _decode_path(record[3:])
        changed.add(path)
        if status == b"??":
            untracked.add(path)
        if status[:1] in {b"R", b"C"} or status[1:2] in {b"R", b"C"}:
            index += 1
            if index >= len(records):
                raise DomainError(ErrorCode.PATCH_REJECTED, "Git 重命名状态不完整")
            original_path = _decode_path(records[index])
            changed.add(original_path)
        index += 1
    return tuple(sorted(changed)), tuple(sorted(untracked))


def _decode_path(data: bytes) -> str:
    try:
        return validate_repository_path(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DomainError(ErrorCode.PATCH_REJECTED, "Git 状态包含不安全路径") from exc


def _assert_binding(lease: WorkspaceLease, spec: AgentExecutionSpec) -> None:
    if (
        lease.task_id != spec.task_id
        or lease.agent_id != spec.agent_id
        or lease.execution_spec_id != spec.spec_id
        or lease.execution_spec_version != spec.version
        or lease.base_commit != spec.base_commit
        or lease.write_scope != spec.write_scope
    ):
        raise DomainError(ErrorCode.PATCH_REJECTED, "补丁工作区与执行规格不匹配")


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args),
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "Git 补丁操作失败") from exc
    if completed.returncode != 0:
        raise DomainError(
            ErrorCode.WORKTREE_OPERATION_FAILED,
            "Git 补丁命令返回非零状态",
            details={"stderr": completed.stderr.decode("utf-8", errors="replace")[:2_000]},
        )
    return completed
