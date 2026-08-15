"""Integrity, syntax, base, path-scope, and sensitive-file patch guards."""

import subprocess
from hashlib import sha256
from pathlib import Path, PurePosixPath

from pydantic import Field

from evoweave_ds.domain.artifacts import PatchArtifact
from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import ArtifactKind
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.integration_models import GuardedPatch
from evoweave_ds.domain.ports import ArtifactStore
from evoweave_ds.domain.validation import path_is_within_scopes, validate_repository_path


class PatchGuardPolicy(DomainModel):
    max_patch_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    forbidden_exact_paths: tuple[str, ...] = (".env",)
    forbidden_prefixes: tuple[str, ...] = (".git", ".runtime")
    forbidden_suffixes: tuple[str, ...] = (".pem", ".key", ".p12", ".pfx")
    forbidden_basenames: tuple[str, ...] = (
        "credentials.json",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    )


class PatchGuard:
    def __init__(
        self,
        artifact_store: ArtifactStore,
        policy: PatchGuardPolicy | None = None,
    ) -> None:
        self._artifact_store = artifact_store
        self._policy = policy or PatchGuardPolicy()

    def inspect(
        self,
        patch: PatchArtifact,
        *,
        expected_base_commit: str,
        write_scope: tuple[str, ...],
        worktree_root: Path | str,
    ) -> GuardedPatch:
        if patch.base_commit != expected_base_commit:
            raise DomainError(
                ErrorCode.PATCH_BASE_MISMATCH,
                "补丁 base commit 与集成基线不一致",
                details={"artifact_id": str(patch.ref.artifact_id)},
            )
        if patch.ref.kind is not ArtifactKind.PATCH or patch.ref.media_type != "text/x-diff":
            raise DomainError(ErrorCode.PATCH_REJECTED, "补丁产物类型或媒体类型无效")
        persisted_ref = self._artifact_store.get_ref(patch.ref.artifact_id)
        if persisted_ref != patch.ref:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "补丁引用与产物库不一致")
        content = self._artifact_store.get_bytes(patch.ref.artifact_id)
        if len(content) > self._policy.max_patch_bytes:
            raise DomainError(ErrorCode.PATCH_REJECTED, "补丁内容超过集成上限")
        if len(content) != patch.ref.size_bytes or sha256(content).hexdigest() != patch.ref.sha256:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "补丁内容身份校验失败")

        root = Path(worktree_root).resolve(strict=True)
        parsed_paths = _parse_patch_paths(root, content)
        if set(parsed_paths) != set(patch.changed_paths):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_ERROR,
                "补丁声明 changed_paths 与实际 diff 不一致",
            )
        for path in parsed_paths:
            if not path_is_within_scopes(path, write_scope):
                raise DomainError(
                    ErrorCode.PATCH_REJECTED,
                    f"补丁路径超出任务 write_scope：{path}",
                )
            if self._is_sensitive(path):
                raise DomainError(ErrorCode.PATCH_REJECTED, f"补丁触及敏感路径：{path}")
        self.check_applies(root, content)
        return GuardedPatch(artifact=patch, parsed_paths=parsed_paths)

    def content_for(self, guarded: GuardedPatch) -> bytes:
        content = self._artifact_store.get_bytes(guarded.artifact.ref.artifact_id)
        if sha256(content).hexdigest() != guarded.artifact.ref.sha256:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "补丁在集成期间发生变化")
        return content

    @staticmethod
    def check_applies(worktree_root: Path | str, content: bytes) -> None:
        root = Path(worktree_root).resolve(strict=True)
        completed = _git_apply(
            root,
            ("apply", "--check", "--binary", "--whitespace=error-all", "-"),
            content,
        )
        if completed.returncode != 0:
            raise DomainError(
                ErrorCode.PATCH_CONFLICT,
                "补丁无法干净应用到当前集成状态",
                details={"stderr": completed.stderr.decode("utf-8", errors="replace")[:2_000]},
            )

    def _is_sensitive(self, path: str) -> bool:
        pure = PurePosixPath(path)
        basename = pure.name.casefold()
        lowered = path.casefold()
        if lowered in {item.casefold() for item in self._policy.forbidden_exact_paths}:
            return True
        if basename in {item.casefold() for item in self._policy.forbidden_basenames}:
            return True
        if any(basename.endswith(item.casefold()) for item in self._policy.forbidden_suffixes):
            return True
        return any(
            lowered == prefix.casefold() or lowered.startswith(f"{prefix.casefold()}/")
            for prefix in self._policy.forbidden_prefixes
        )


def _parse_patch_paths(root: Path, content: bytes) -> tuple[str, ...]:
    completed = _git_apply(root, ("apply", "--numstat", "-z", "--binary", "-"), content)
    if completed.returncode != 0:
        raise DomainError(
            ErrorCode.PATCH_REJECTED,
            "补丁语法无法由 Git 解析",
            details={"stderr": completed.stderr.decode("utf-8", errors="replace")[:2_000]},
        )
    records = completed.stdout.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            raise DomainError(ErrorCode.PATCH_REJECTED, "无法解析补丁路径统计")
        if fields[2]:
            paths.append(_decode_path(fields[2]))
            continue
        if index + 1 >= len(records):
            raise DomainError(ErrorCode.PATCH_REJECTED, "补丁重命名路径记录不完整")
        paths.append(_decode_path(records[index]))
        paths.append(_decode_path(records[index + 1]))
        index += 2
    if not paths:
        raise DomainError(ErrorCode.PATCH_EMPTY, "补丁没有可应用路径")
    return tuple(sorted(set(paths)))


def _decode_path(value: bytes) -> str:
    try:
        return validate_repository_path(value.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DomainError(ErrorCode.PATCH_REJECTED, "补丁包含不安全路径") from exc


def _git_apply(
    root: Path,
    args: tuple[str, ...],
    content: bytes,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ("git", "-C", str(root), *args),
            input=content,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DomainError(ErrorCode.WORKTREE_OPERATION_FAILED, "Git 补丁检查失败") from exc
