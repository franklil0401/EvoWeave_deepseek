"""Preflight and serially integrate patches without touching the user branch."""

from evoweave_ds.domain.artifacts import PatchArtifact
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import RunId, TaskId
from evoweave_ds.domain.integration_models import GuardedPatch, IntegrationWorkspaceState
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.integration.conflict_detector import PatchConflictDetector
from evoweave_ds.integration.integration_workspace import IntegrationWorkspaceManager
from evoweave_ds.integration.patch_applier import PatchApplier
from evoweave_ds.integration.patch_guard import PatchGuard
from evoweave_ds.integration.planner import PatchIntegrationPlanner


class PatchIntegrationService:
    def __init__(
        self,
        *,
        manager: IntegrationWorkspaceManager,
        guard: PatchGuard,
        applier: PatchApplier,
        planner: PatchIntegrationPlanner | None = None,
        conflict_detector: PatchConflictDetector | None = None,
    ) -> None:
        self._manager = manager
        self._guard = guard
        self._applier = applier
        self._planner = planner or PatchIntegrationPlanner()
        self._conflict_detector = conflict_detector or PatchConflictDetector()

    def integrate(
        self,
        *,
        run_id: RunId,
        base_commit: str,
        patches: tuple[PatchArtifact, ...],
        task_specs: tuple[TaskSpec, ...],
    ) -> IntegrationWorkspaceState:
        state = self._manager.create(run_id=run_id, base_commit=base_commit)
        try:
            ordered = self._planner.order(patches, task_specs)
            spec_by_task = {spec.task_id: spec for spec in task_specs}
            guarded: list[GuardedPatch] = []
            for patch in ordered:
                spec = spec_by_task[patch.task_id]
                self._validate_binding(patch, spec, base_commit)
                guarded.append(
                    self._guard.inspect(
                        patch,
                        expected_base_commit=base_commit,
                        write_scope=spec.write_scope,
                        worktree_root=state.worktree_path,
                    )
                )
            conflicts = self._conflict_detector.detect(tuple(guarded))
            if conflicts:
                raise DomainError(
                    ErrorCode.PATCH_CONFLICT,
                    "补丁集合存在实际写集合冲突",
                    details={"conflicts": [item.model_dump(mode="json") for item in conflicts]},
                )
            for item in guarded:
                content = self._guard.content_for(item)
                state = self._applier.apply(state, item, content)
                self._manager.save(state)
            return state
        except Exception as exc:
            while state.applied_patches:
                state = self._applier.rollback_latest(state)
                self._manager.save(state)
            reason = exc.message if isinstance(exc, DomainError) else type(exc).__name__
            self._manager.mark_failed(state.integration_id, reason)
            raise

    @staticmethod
    def _validate_binding(
        patch: PatchArtifact,
        spec: TaskSpec,
        base_commit: str,
    ) -> None:
        if patch.task_id != spec.task_id:
            raise DomainError(ErrorCode.INVALID_SPEC, "补丁与 TaskSpec 任务不一致")
        if patch.base_commit != spec.base_commit or patch.base_commit != base_commit:
            raise DomainError(ErrorCode.PATCH_BASE_MISMATCH, "补丁、任务与集成基线不一致")
        if not spec.write_scope:
            raise DomainError(ErrorCode.PATCH_REJECTED, "无写权限任务不能提交补丁")


def write_scopes_by_task(task_specs: tuple[TaskSpec, ...]) -> dict[TaskId, tuple[str, ...]]:
    return {spec.task_id: spec.write_scope for spec in task_specs}
