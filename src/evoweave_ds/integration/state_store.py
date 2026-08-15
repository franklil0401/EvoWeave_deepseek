"""Atomic state persistence for integration worktrees."""

import os
from pathlib import Path

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import IntegrationId
from evoweave_ds.domain.integration_models import IntegrationWorkspaceState


class JsonIntegrationStateStore:
    def __init__(self, state_root: Path | str) -> None:
        self._root = Path(state_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, state: IntegrationWorkspaceState) -> None:
        destination = self._path_for(state.integration_id)
        if destination.exists():
            current = IntegrationWorkspaceState.model_validate_json(destination.read_bytes())
            if current == state:
                return
            if (
                current.integration_id != state.integration_id
                or current.run_id != state.run_id
                or current.repository_root != state.repository_root
                or current.worktree_path != state.worktree_path
                or current.branch_name != state.branch_name
                or current.base_commit != state.base_commit
            ):
                raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "集成状态身份字段不可修改")
            if state.version != current.version + 1:
                raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "集成状态版本必须连续递增")
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, destination)

    def get(self, integration_id: IntegrationId) -> IntegrationWorkspaceState:
        try:
            return IntegrationWorkspaceState.model_validate_json(
                self._path_for(integration_id).read_bytes()
            )
        except FileNotFoundError as exc:
            raise DomainError(ErrorCode.INTEGRATION_STATE_INVALID, "找不到集成工作区状态") from exc

    def list_all(self) -> tuple[IntegrationWorkspaceState, ...]:
        states = [
            IntegrationWorkspaceState.model_validate_json(path.read_bytes())
            for path in self._root.glob("integration_*.json")
        ]
        return tuple(sorted(states, key=lambda item: str(item.integration_id)))

    def _path_for(self, integration_id: IntegrationId) -> Path:
        path = (self._root / f"{integration_id}.json").resolve()
        if path.parent != self._root:
            raise DomainError(ErrorCode.WORKSPACE_ACCESS_DENIED, "集成状态路径越界")
        return path
