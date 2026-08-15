"""Explicit latest-patch rollback through the integration state manager."""

from evoweave_ds.domain.identifiers import IntegrationId
from evoweave_ds.domain.integration_models import IntegrationWorkspaceState
from evoweave_ds.integration.integration_workspace import IntegrationWorkspaceManager
from evoweave_ds.integration.patch_applier import PatchApplier


class IntegrationRollback:
    def __init__(
        self,
        manager: IntegrationWorkspaceManager,
        applier: PatchApplier,
    ) -> None:
        self._manager = manager
        self._applier = applier

    def latest(self, integration_id: IntegrationId) -> IntegrationWorkspaceState:
        state = self._manager.get(integration_id)
        revised = self._applier.rollback_latest(state)
        self._manager.save(revised)
        return revised
