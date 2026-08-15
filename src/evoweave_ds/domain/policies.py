"""Deterministic policy contracts for graph growth and runtime limits."""

from pydantic import Field

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import PolicyViolationCode, TaskStatus
from evoweave_ds.domain.graph_models import GraphSnapshot
from evoweave_ds.domain.identifiers import TaskId


class GraphPolicy(DomainModel):
    max_nodes: int = Field(default=64, ge=1, le=10_000)
    max_concurrent_tasks: int = Field(default=6, ge=1, le=1_000)
    max_attempts_per_task: int = Field(default=3, ge=1, le=100)
    max_tasks_per_decision: int = Field(default=8, ge=1, le=1_000)
    max_decisions: int = Field(default=256, ge=1, le=100_000)
    max_no_progress_decisions: int = Field(default=8, ge=1, le=1_000)
    task_lease_seconds: int = Field(default=1_800, ge=1, le=86_400)


class PolicyViolation(DomainModel):
    code: PolicyViolationCode
    message: str = Field(min_length=1, max_length=2_000)
    task_id: TaskId | None = None


class PolicyDecision(DomainModel):
    allowed: bool
    violations: tuple[PolicyViolation, ...] = ()


def evaluate_graph_policy(snapshot: GraphSnapshot, policy: GraphPolicy) -> PolicyDecision:
    """Evaluate deterministic graph limits without invoking a model."""

    violations: list[PolicyViolation] = []
    if len(snapshot.nodes) > policy.max_nodes:
        violations.append(
            PolicyViolation(
                code=PolicyViolationCode.TOO_MANY_NODES,
                message=f"节点数 {len(snapshot.nodes)} 超过上限 {policy.max_nodes}",
            )
        )
    running = [
        node for node in snapshot.nodes if node.status in {TaskStatus.LEASED, TaskStatus.RUNNING}
    ]
    if len(running) > policy.max_concurrent_tasks:
        violations.append(
            PolicyViolation(
                code=PolicyViolationCode.TOO_MANY_RUNNING_TASKS,
                message=f"并发任务数 {len(running)} 超过上限 {policy.max_concurrent_tasks}",
            )
        )
    for node in snapshot.nodes:
        if node.attempts > policy.max_attempts_per_task:
            violations.append(
                PolicyViolation(
                    code=PolicyViolationCode.RETRY_LIMIT_EXCEEDED,
                    message=f"任务尝试次数 {node.attempts} 超过上限",
                    task_id=node.task_id,
                )
            )
    return PolicyDecision(allowed=not violations, violations=tuple(violations))
