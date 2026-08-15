"""Deterministic ready-task scheduling with slots, leases, and write-set exclusion."""

from datetime import datetime, timedelta

from pydantic import Field, model_validator

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.base import DomainModel, utc_now
from evoweave_ds.domain.enums import TaskLeaseStatus, TaskStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId, TaskLeaseId
from evoweave_ds.domain.policies import GraphPolicy
from evoweave_ds.domain.validation import path_is_within_scopes
from evoweave_ds.orchestration.task_graph import TaskGraph


class TaskLease(DomainModel):
    lease_id: TaskLeaseId
    run_id: RunId
    task_id: TaskId
    agent_id: AgentId
    execution_spec_id: SpecId
    execution_spec_version: int = Field(ge=1)
    write_scope: tuple[str, ...] = ()
    status: TaskLeaseStatus = TaskLeaseStatus.ACTIVE
    leased_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime

    @model_validator(mode="after")
    def validate_expiry(self) -> "TaskLease":
        if self.expires_at <= self.leased_at:
            raise ValueError("任务租约过期时间必须晚于创建时间")
        return self


class AgentAllocationDecision(DomainModel):
    allocation_id: SpecId
    run_id: RunId
    graph_version: int = Field(ge=1)
    ready_task_ids: tuple[TaskId, ...]
    selected_task_ids: tuple[TaskId, ...]
    available_slots: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=2_000)


class Scheduler:
    def __init__(self, policy: GraphPolicy | None = None) -> None:
        self._policy = policy or GraphPolicy()

    def select_ready(
        self,
        graph: TaskGraph,
        active_leases: tuple[TaskLease, ...] = (),
    ) -> tuple[TaskId, ...]:
        active = tuple(lease for lease in active_leases if lease.status is TaskLeaseStatus.ACTIVE)
        slots = max(0, self._policy.max_concurrent_tasks - len(active))
        if slots == 0:
            return ()
        occupied_scopes = [lease.write_scope for lease in active if lease.write_scope]
        selected: list[TaskId] = []
        for node in sorted(graph.snapshot.nodes, key=lambda item: str(item.task_id)):
            if node.status is not TaskStatus.READY:
                continue
            candidate_scope = graph.spec_for(node.task_id).write_scope
            if candidate_scope and any(
                write_scopes_overlap(candidate_scope, occupied) for occupied in occupied_scopes
            ):
                continue
            selected.append(node.task_id)
            if candidate_scope:
                occupied_scopes.append(candidate_scope)
            if len(selected) >= slots:
                break
        return tuple(selected)

    def plan(
        self,
        graph: TaskGraph,
        active_leases: tuple[TaskLease, ...] = (),
    ) -> AgentAllocationDecision:
        ready = tuple(
            node.task_id
            for node in sorted(graph.snapshot.nodes, key=lambda item: str(item.task_id))
            if node.status is TaskStatus.READY
        )
        active_count = sum(lease.status is TaskLeaseStatus.ACTIVE for lease in active_leases)
        available_slots = max(0, self._policy.max_concurrent_tasks - active_count)
        selected = self.select_ready(graph, active_leases)
        return AgentAllocationDecision(
            allocation_id=SpecId.new(),
            run_id=graph.snapshot.run_id,
            graph_version=graph.snapshot.version,
            ready_task_ids=ready,
            selected_task_ids=selected,
            available_slots=available_slots,
            reason="实例数由就绪任务、并发槽位和写集合独立性共同决定",
        )

    def lease(
        self,
        graph: TaskGraph,
        execution_spec: AgentExecutionSpec,
        *,
        now: datetime | None = None,
    ) -> TaskLease:
        node = graph.node_for(execution_spec.task_id)
        task_spec = graph.spec_for(execution_spec.task_id)
        if node.status is not TaskStatus.READY:
            raise DomainError(ErrorCode.INVALID_STATE_TRANSITION, "只能租用 READY 任务")
        if execution_spec.task_spec_id != task_spec.spec_id or (
            execution_spec.task_spec_version != task_spec.version
        ):
            raise DomainError(ErrorCode.INVALID_SPEC, "AgentExecutionSpec 未绑定当前 TaskSpec")
        leased_at = now or utc_now()
        lease = TaskLease(
            lease_id=TaskLeaseId.new(),
            run_id=graph.snapshot.run_id,
            task_id=execution_spec.task_id,
            agent_id=execution_spec.agent_id,
            execution_spec_id=execution_spec.spec_id,
            execution_spec_version=execution_spec.version,
            write_scope=execution_spec.write_scope,
            leased_at=leased_at,
            expires_at=leased_at + timedelta(seconds=self._policy.task_lease_seconds),
        )
        graph.transition(execution_spec.task_id, TaskStatus.LEASED)
        return lease

    def expire(
        self,
        graph: TaskGraph,
        lease: TaskLease,
        *,
        now: datetime | None = None,
    ) -> TaskLease:
        current_time = now or utc_now()
        if lease.status is not TaskLeaseStatus.ACTIVE or current_time < lease.expires_at:
            return lease
        node = graph.node_for(lease.task_id)
        if node.status is TaskStatus.LEASED:
            graph.transition(lease.task_id, TaskStatus.READY)
        elif node.status is TaskStatus.RUNNING:
            graph.transition(lease.task_id, TaskStatus.FAILED)
        return lease.model_copy(update={"status": TaskLeaseStatus.EXPIRED})


def write_scopes_overlap(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    return any(
        path_is_within_scopes(path, (other,)) or path_is_within_scopes(other, (path,))
        for path in first
        for other in second
    )
