"""Best-effort cleanup wrapper that preserves recoverable lease state."""

from evoweave_ds.domain.identifiers import WorkspaceId
from evoweave_ds.domain.workspace_models import WorkspaceLease
from evoweave_ds.workspaces.manager import WorkspaceManager


class WorkspaceCleanup:
    def __init__(self, manager: WorkspaceManager) -> None:
        self._manager = manager

    def release(self, workspace_id: WorkspaceId) -> WorkspaceLease:
        return self._manager.release(workspace_id)
