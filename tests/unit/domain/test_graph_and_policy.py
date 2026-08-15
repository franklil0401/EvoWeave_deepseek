"""Tests for graph validity, task state transitions, and deterministic policy."""

import pytest
from pydantic import ValidationError

from evoweave_ds.domain.enums import PolicyViolationCode, TaskRelation, TaskStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.graph_models import GraphSnapshot, TaskEdge, TaskNode
from evoweave_ds.domain.identifiers import GraphId, RunId, TaskId
from evoweave_ds.domain.policies import GraphPolicy, evaluate_graph_policy


def _node(status: TaskStatus = TaskStatus.CREATED, attempts: int = 0) -> TaskNode:
    return TaskNode(
        task_id=TaskId.new(),
        task_spec_version=1,
        status=status,
        attempts=attempts,
    )


def test_independent_root_tasks_are_valid() -> None:
    snapshot = GraphSnapshot(
        graph_id=GraphId.new(), run_id=RunId.new(), version=1, nodes=(_node(), _node())
    )
    assert len(snapshot.nodes) == 2


def test_graph_rejects_missing_edge_endpoint() -> None:
    node = _node()
    with pytest.raises(ValidationError, match="不存在的节点"):
        GraphSnapshot(
            graph_id=GraphId.new(),
            run_id=RunId.new(),
            version=1,
            nodes=(node,),
            edges=(
                TaskEdge(
                    source=node.task_id,
                    target=TaskId.new(),
                    relation=TaskRelation.DEPENDS_ON,
                ),
            ),
        )


def test_graph_rejects_duplicate_task_identifier() -> None:
    node = _node()
    with pytest.raises(ValidationError, match="重复节点"):
        GraphSnapshot(
            graph_id=GraphId.new(),
            run_id=RunId.new(),
            version=1,
            nodes=(node, node.model_copy()),
        )


def test_graph_rejects_dependency_cycle() -> None:
    first, second = _node(), _node()
    with pytest.raises(ValidationError, match="循环"):
        GraphSnapshot(
            graph_id=GraphId.new(),
            run_id=RunId.new(),
            version=1,
            nodes=(first, second),
            edges=(
                TaskEdge(
                    source=first.task_id,
                    target=second.task_id,
                    relation=TaskRelation.DEPENDS_ON,
                ),
                TaskEdge(
                    source=second.task_id,
                    target=first.task_id,
                    relation=TaskRelation.DEPENDS_ON,
                ),
            ),
        )


def test_task_state_machine_increments_attempt_on_running() -> None:
    node = _node()
    node = node.transition_to(TaskStatus.READY)
    node = node.transition_to(TaskStatus.LEASED)
    node = node.transition_to(TaskStatus.RUNNING)
    assert node.status is TaskStatus.RUNNING
    assert node.attempts == 1


def test_terminal_task_state_cannot_transition() -> None:
    node = _node(TaskStatus.SUCCEEDED)
    with pytest.raises(DomainError) as error:
        node.transition_to(TaskStatus.READY)
    assert error.value.code is ErrorCode.INVALID_STATE_TRANSITION


def test_graph_policy_reports_all_limit_violations() -> None:
    first = _node(TaskStatus.RUNNING, attempts=4)
    second = _node(TaskStatus.LEASED)
    snapshot = GraphSnapshot(
        graph_id=GraphId.new(),
        run_id=RunId.new(),
        version=1,
        nodes=(first, second),
    )
    decision = evaluate_graph_policy(
        snapshot,
        GraphPolicy(max_nodes=1, max_concurrent_tasks=1, max_attempts_per_task=3),
    )
    assert not decision.allowed
    assert {violation.code for violation in decision.violations} == {
        PolicyViolationCode.TOO_MANY_NODES,
        PolicyViolationCode.TOO_MANY_RUNNING_TASKS,
        PolicyViolationCode.RETRY_LIMIT_EXCEEDED,
    }
