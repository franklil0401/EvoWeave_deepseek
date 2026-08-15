"""Atomic JSON persistence for recoverable workspace lease state."""

import os
from pathlib import Path

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import WorkspaceId
from evoweave_ds.domain.workspace_models import WorkspaceLease


class JsonWorkspaceLeaseStore:
    def __init__(self, state_root: Path | str) -> None:
        self._root = Path(state_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, lease: WorkspaceLease) -> None:
        destination = self._path_for(lease.workspace_id)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(lease.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, destination)

    def get(self, workspace_id: WorkspaceId) -> WorkspaceLease:
        path = self._path_for(workspace_id)
        try:
            return WorkspaceLease.model_validate_json(path.read_bytes())
        except FileNotFoundError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_LEASE_NOT_FOUND,
                f"找不到工作区租约：{workspace_id}",
            ) from exc

    def list_all(self) -> tuple[WorkspaceLease, ...]:
        leases = [
            WorkspaceLease.model_validate_json(path.read_bytes())
            for path in self._root.glob("workspace_*.json")
        ]
        return tuple(sorted(leases, key=lambda item: str(item.workspace_id)))

    def _path_for(self, workspace_id: WorkspaceId) -> Path:
        path = (self._root / f"{workspace_id}.json").resolve()
        if path.parent != self._root:
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "租约状态路径越界")
        return path
