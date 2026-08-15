"""Versioned task graph contracts and graph invariants."""

from collections import defaultdict, deque
from datetime import datetime

from pydantic import Field, model_validator

from evoweave_ds.domain.base import DomainModel, utc_now
from evoweave_ds.domain.enums import TaskRelation, TaskStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import GraphId, RunId, TaskId

_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.READY, TaskStatus.CANCELLED, TaskStatus.BLOCKED}),
    TaskStatus.READY: frozenset({TaskStatus.LEASED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}),
    TaskStatus.LEASED: frozenset({TaskStatus.RUNNING, TaskStatus.READY, TaskStatus.CANCELLED}),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}
    ),
    TaskStatus.FAILED: frozenset({TaskStatus.READY, TaskStatus.CANCELLED, TaskStatus.BLOCKED}),
    TaskStatus.BLOCKED: frozenset({TaskStatus.READY, TaskStatus.CANCELLED}),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


class TaskNode(DomainModel):
    task_id: TaskId
    task_spec_version: int = Field(ge=1)
    status: TaskStatus = TaskStatus.CREATED
    attempts: int = Field(default=0, ge=0)

    def transition_to(self, target: TaskStatus) -> "TaskNode":
        """Return a new node if and only if the transition is legal."""

        if target not in _ALLOWED_TRANSITIONS[self.status]:
            raise DomainError(
                ErrorCode.INVALID_STATE_TRANSITION,
                f"非法任务状态转换：{self.status} -> {target}",
                details={"task_id": str(self.task_id)},
            )
        attempts = self.attempts + (1 if target is TaskStatus.RUNNING else 0)
        return self.model_copy(update={"status": target, "attempts": attempts})


class TaskEdge(DomainModel):
    source: TaskId
    target: TaskId
    relation: TaskRelation

    @model_validator(mode="after")
    def reject_self_edge(self) -> "TaskEdge":
        if self.source == self.target:
            raise ValueError("任务边不能指向自身")
        return self


class GraphSnapshot(DomainModel):
    graph_id: GraphId
    run_id: RunId
    version: int = Field(ge=1)
    nodes: tuple[TaskNode, ...]
    edges: tuple[TaskEdge, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_graph(self) -> "GraphSnapshot":
        node_ids = [node.task_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("任务图不能包含重复节点")
        node_set = set(node_ids)

        edge_keys = [(edge.source, edge.target, edge.relation) for edge in self.edges]
        if len(set(edge_keys)) != len(edge_keys):
            raise ValueError("任务图不能包含重复边")
        for edge in self.edges:
            if edge.source not in node_set or edge.target not in node_set:
                raise ValueError("任务边引用了不存在的节点")

        dependency_pairs = [
            (edge.source, edge.target)
            for edge in self.edges
            if edge.relation is TaskRelation.DEPENDS_ON
        ]
        self._assert_acyclic(node_set, dependency_pairs)
        return self

    @staticmethod
    def _assert_acyclic(
        node_ids: set[TaskId],
        dependency_pairs: list[tuple[TaskId, TaskId]],
    ) -> None:
        adjacency: dict[TaskId, list[TaskId]] = defaultdict(list)
        indegree: dict[TaskId, int] = {task_id: 0 for task_id in node_ids}
        for prerequisite, dependent in dependency_pairs:
            adjacency[prerequisite].append(dependent)
            indegree[dependent] += 1
        queue = deque(task_id for task_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            task_id = queue.popleft()
            visited += 1
            for dependent in adjacency[task_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if visited != len(node_ids):
            raise ValueError("任务依赖图不能包含循环")
