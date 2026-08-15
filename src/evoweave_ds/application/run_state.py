"""Atomic run-manifest persistence and legal lifecycle transitions."""

import os
from pathlib import Path

from evoweave_ds.domain.base import utc_now
from evoweave_ds.domain.enums import RunStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import ArtifactId, RunId
from evoweave_ds.domain.run_models import RunManifest

_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.INITIALIZED: frozenset(
        {RunStatus.ANALYZED, RunStatus.RUNNING, RunStatus.WAITING_FOR_INPUT, RunStatus.FAILED}
    ),
    RunStatus.ANALYZED: frozenset(
        {RunStatus.RUNNING, RunStatus.WAITING_FOR_INPUT, RunStatus.FAILED}
    ),
    RunStatus.RUNNING: frozenset(
        {RunStatus.WAITING_FOR_INPUT, RunStatus.COMPLETED, RunStatus.FAILED}
    ),
    RunStatus.WAITING_FOR_INPUT: frozenset({RunStatus.RUNNING, RunStatus.FAILED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
}


class JsonRunStateStore:
    def __init__(self, state_root: Path | str) -> None:
        self._root = Path(state_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def create(self, manifest: RunManifest) -> None:
        path = self._path_for(manifest.run_id)
        if path.exists():
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "run_id 已存在")
        self._write(path, manifest)

    def save(self, manifest: RunManifest) -> None:
        current = self.get(manifest.run_id)
        if manifest.version != current.version + 1:
            raise DomainError(ErrorCode.INVALID_STATE_TRANSITION, "运行版本必须连续递增")
        if (
            manifest.status != current.status
            and manifest.status not in _ALLOWED_TRANSITIONS[current.status]
        ):
            raise DomainError(ErrorCode.INVALID_STATE_TRANSITION, "非法运行状态转换")
        if manifest.change_spec != current.change_spec or manifest.created_at != current.created_at:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "运行身份字段不可修改")
        self._write(self._path_for(manifest.run_id), manifest)

    def get(self, run_id: RunId) -> RunManifest:
        try:
            return RunManifest.model_validate_json(self._path_for(run_id).read_bytes())
        except FileNotFoundError as exc:
            raise DomainError(ErrorCode.ARTIFACT_NOT_FOUND, "找不到运行状态") from exc

    def list_all(self) -> tuple[RunManifest, ...]:
        manifests = [
            RunManifest.model_validate_json(path.read_bytes())
            for path in self._root.glob("run_*.json")
        ]
        return tuple(sorted(manifests, key=lambda item: item.created_at, reverse=True))

    def transition(
        self,
        run_id: RunId,
        status: RunStatus,
        *,
        message: str,
        error_code: ErrorCode | None = None,
        repository_profile_artifact_id: ArtifactId | None = None,
        final_patch_artifact_id: ArtifactId | None = None,
        validation_report_artifact_id: ArtifactId | None = None,
    ) -> RunManifest:
        current = self.get(run_id)
        revised = current.model_copy(
            update={
                "status": status,
                "message": message,
                "error_code": error_code,
                "repository_profile_artifact_id": (
                    repository_profile_artifact_id or current.repository_profile_artifact_id
                ),
                "final_patch_artifact_id": (
                    final_patch_artifact_id or current.final_patch_artifact_id
                ),
                "validation_report_artifact_id": (
                    validation_report_artifact_id or current.validation_report_artifact_id
                ),
                "updated_at": utc_now(),
                "version": current.version + 1,
            }
        )
        self.save(revised)
        return revised

    def _path_for(self, run_id: RunId) -> Path:
        path = (self._root / f"{run_id}.json").resolve()
        if path.parent != self._root:
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "运行状态路径越界")
        return path

    @staticmethod
    def _write(path: Path, manifest: RunManifest) -> None:
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
