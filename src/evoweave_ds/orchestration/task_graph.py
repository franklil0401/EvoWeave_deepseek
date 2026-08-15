"""Transactional in-memory task graph with immutable versioned snapshots."""

from evoweave_ds.domain.enums import TaskRelation, TaskStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.graph_models import GraphSnapshot, TaskEdge, TaskNode
from evoweave_ds.domain.identifiers import GraphId, RunId, TaskId
from evoweave_ds.domain.policies import GraphPolicy, evaluate_graph_policy
from evoweave_ds.domain.task_spec import TaskSpec


class TaskGraph:
    def __init__(
        self,
        *,
        snapshot: GraphSnapshot,
        task_specs: tuple[TaskSpec, ...],
        policy: GraphPolicy | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._history = [snapshot]
        self._specs = {spec.task_id: spec for spec in task_specs}
        self._records = [(snapshot, self._ordered_specs(self._specs))]
        self._policy = policy or GraphPolicy()
        if set(self._specs) != {node.task_id for node in snapshot.nodes}:
            raise ValueError("任务规格集合必须与图节点一一对应")
        for node in snapshot.nodes:
            if self._specs[node.task_id].version != node.task_spec_version:
                raise ValueError("节点记录的 TaskSpec 版本与实际规格不一致")

    @classmethod
    def create(
        cls,
        *,
        run_id: RunId,
        task_specs: tuple[TaskSpec, ...],
        policy: GraphPolicy | None = None,
    ) -> "TaskGraph":
        if not task_specs:
            raise ValueError("初始任务图至少需要一个任务")
        graph_id = GraphId.new()
        snapshot = GraphSnapshot(
            graph_id=graph_id,
            run_id=run_id,
            version=1,
            nodes=tuple(
                TaskNode(task_id=spec.task_id, task_spec_version=spec.version)
                for spec in task_specs
            ),
            edges=_dependency_edges(task_specs),
        )
        graph = cls(snapshot=snapshot, task_specs=task_specs, policy=policy)
        graph.refresh_ready()
        return graph

    @property
    def snapshot(self) -> GraphSnapshot:
        return self._snapshot

    @property
    def task_specs(self) -> tuple[TaskSpec, ...]:
        return tuple(self._specs[key] for key in sorted(self._specs, key=str))

    @property
    def history(self) -> tuple[GraphSnapshot, ...]:
        return tuple(self._history)

    @property
    def version_records(self) -> tuple[tuple[GraphSnapshot, tuple[TaskSpec, ...]], ...]:
        return tuple(self._records)

    def spec_for(self, task_id: TaskId) -> TaskSpec:
        try:
            return self._specs[task_id]
        except KeyError as exc:
            raise DomainError(ErrorCode.INVALID_GRAPH, f"任务不存在：{task_id}") from exc

    def node_for(self, task_id: TaskId) -> TaskNode:
        try:
            return next(node for node in self._snapshot.nodes if node.task_id == task_id)
        except StopIteration as exc:
            raise DomainError(ErrorCode.INVALID_GRAPH, f"任务节点不存在：{task_id}") from exc

    def add_tasks(
        self,
        task_specs: tuple[TaskSpec, ...],
        *,
        extra_edges: tuple[TaskEdge, ...] = (),
    ) -> GraphSnapshot:
        if not task_specs:
            raise ValueError("新增任务不能为空")
        existing_ids = set(self._specs)
        incoming_ids = [spec.task_id for spec in task_specs]
        if existing_ids.intersection(incoming_ids) or len(set(incoming_ids)) != len(incoming_ids):
            raise DomainError(ErrorCode.INVALID_GRAPH, "新增任务 ID 重复")
        nodes = self._snapshot.nodes + tuple(
            TaskNode(task_id=spec.task_id, task_spec_version=spec.version) for spec in task_specs
        )
        edges = self._snapshot.edges + _dependency_edges(task_specs) + extra_edges
        candidate = self._candidate(nodes=nodes, edges=edges)
        new_specs = {**self._specs, **{spec.task_id: spec for spec in task_specs}}
        self._commit(candidate, new_specs)
        self.refresh_ready()
        return self._snapshot

    def replace_spec(self, replacement: TaskSpec) -> GraphSnapshot:
        current = self.spec_for(replacement.task_id)
        if replacement.version != current.version + 1:
            raise DomainError(ErrorCode.INVALID_SPEC, "替换 TaskSpec 必须严格增加一个版本")
        nodes = tuple(
            node.model_copy(update={"task_spec_version": replacement.version})
            if node.task_id == replacement.task_id
            else node
            for node in self._snapshot.nodes
        )
        candidate = self._candidate(nodes=nodes)
        new_specs = dict(self._specs)
        new_specs[replacement.task_id] = replacement
        self._commit(candidate, new_specs)
        return self._snapshot

    def transition(self, task_id: TaskId, target: TaskStatus) -> GraphSnapshot:
        nodes = tuple(
            node.transition_to(target) if node.task_id == task_id else node
            for node in self._snapshot.nodes
        )
        if nodes == self._snapshot.nodes:
            raise DomainError(ErrorCode.INVALID_GRAPH, f"任务不存在：{task_id}")
        self._commit(self._candidate(nodes=nodes), dict(self._specs))
        if target in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            self.refresh_ready()
        return self._snapshot

    def refresh_ready(self) -> GraphSnapshot:
        node_by_id = {node.task_id: node for node in self._snapshot.nodes}
        prerequisites: dict[TaskId, set[TaskId]] = {task_id: set() for task_id in node_by_id}
        for edge in self._snapshot.edges:
            if edge.relation is TaskRelation.DEPENDS_ON:
                prerequisites[edge.target].add(edge.source)
        changed = False
        nodes: list[TaskNode] = []
        for node in self._snapshot.nodes:
            if node.status is TaskStatus.CREATED and all(
                node_by_id[dependency].status is TaskStatus.SUCCEEDED
                for dependency in prerequisites[node.task_id]
            ):
                nodes.append(node.transition_to(TaskStatus.READY))
                changed = True
            else:
                nodes.append(node)
        if changed:
            self._commit(self._candidate(nodes=tuple(nodes)), dict(self._specs))
        return self._snapshot

    def _candidate(
        self,
        *,
        nodes: tuple[TaskNode, ...] | None = None,
        edges: tuple[TaskEdge, ...] | None = None,
    ) -> GraphSnapshot:
        return GraphSnapshot(
            graph_id=self._snapshot.graph_id,
            run_id=self._snapshot.run_id,
            version=self._snapshot.version + 1,
            nodes=nodes if nodes is not None else self._snapshot.nodes,
            edges=edges if edges is not None else self._snapshot.edges,
            created_at=self._snapshot.created_at,
        )

    def _commit(self, candidate: GraphSnapshot, specs: dict[TaskId, TaskSpec]) -> None:
        policy_decision = evaluate_graph_policy(candidate, self._policy)
        if not policy_decision.allowed:
            raise DomainError(
                ErrorCode.POLICY_REJECTED,
                "任务图变更违反 GraphPolicy",
                details={"violations": [item.code.value for item in policy_decision.violations]},
            )
        self._snapshot = candidate
        self._specs = specs
        self._history.append(candidate)
        self._records.append((candidate, self._ordered_specs(specs)))

    @staticmethod
    def _ordered_specs(specs: dict[TaskId, TaskSpec]) -> tuple[TaskSpec, ...]:
        return tuple(specs[key] for key in sorted(specs, key=str))


def _dependency_edges(task_specs: tuple[TaskSpec, ...]) -> tuple[TaskEdge, ...]:
    return tuple(
        TaskEdge(source=dependency, target=spec.task_id, relation=TaskRelation.DEPENDS_ON)
        for spec in task_specs
        for dependency in spec.depends_on
    )
