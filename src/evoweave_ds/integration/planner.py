"""Deterministic dependency order for one patch per completed task."""

from heapq import heappop, heappush

from evoweave_ds.domain.artifacts import PatchArtifact
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import TaskId
from evoweave_ds.domain.task_spec import TaskSpec


class PatchIntegrationPlanner:
    def order(
        self,
        patches: tuple[PatchArtifact, ...],
        task_specs: tuple[TaskSpec, ...],
    ) -> tuple[PatchArtifact, ...]:
        if not patches:
            raise DomainError(ErrorCode.PATCH_EMPTY, "集成补丁集合不能为空")
        patch_by_task = {patch.task_id: patch for patch in patches}
        if len(patch_by_task) != len(patches):
            raise DomainError(ErrorCode.PATCH_CONFLICT, "同一任务不能提供多个补丁")
        spec_by_task = {spec.task_id: spec for spec in task_specs}
        missing = set(patch_by_task).difference(spec_by_task)
        if missing:
            raise DomainError(ErrorCode.INVALID_SPEC, "补丁缺少对应 TaskSpec")

        indegree: dict[TaskId, int] = {task_id: 0 for task_id in patch_by_task}
        dependents: dict[TaskId, list[TaskId]] = {task_id: [] for task_id in patch_by_task}
        for task_id in patch_by_task:
            for dependency in spec_by_task[task_id].depends_on:
                if dependency not in patch_by_task:
                    continue
                indegree[task_id] += 1
                dependents[dependency].append(task_id)
        ready: list[str] = []
        task_by_text = {str(task_id): task_id for task_id in patch_by_task}
        for task_id, degree in indegree.items():
            if degree == 0:
                heappush(ready, str(task_id))
        ordered: list[PatchArtifact] = []
        while ready:
            task_id = task_by_text[heappop(ready)]
            ordered.append(patch_by_task[task_id])
            for dependent in sorted(dependents[task_id], key=str):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heappush(ready, str(dependent))
        if len(ordered) != len(patches):
            raise DomainError(ErrorCode.INVALID_GRAPH, "补丁任务依赖存在循环")
        return tuple(ordered)
