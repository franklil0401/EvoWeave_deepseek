"""The sole control plane for finite decisions, workers, and recoverable graph state."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Protocol

from evoweave_ds.agent_runtime.fallback import revise_execution_spec_for_routing
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import ResultStatus, TaskLeaseStatus, TaskRelation, TaskStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.graph_models import GraphSnapshot, TaskEdge
from evoweave_ds.domain.identifiers import AgentId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelRoutingDecision
from evoweave_ds.domain.policies import GraphPolicy
from evoweave_ds.domain.ports import DecisionLedger, GraphStateStore
from evoweave_ds.domain.task_result import TaskResult
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.orchestration.agent_factory import AgentFactory, CapabilityPlan
from evoweave_ds.orchestration.checkpointing import (
    CheckpointManager,
    OrchestrationCheckpoint,
    ProcessedDecisionRecord,
)
from evoweave_ds.orchestration.control_view import (
    OrchestrationControlView,
    ResultControlSummary,
    TaskControlItem,
    TaskSuggestion,
)
from evoweave_ds.orchestration.decisions import (
    CancelTaskAction,
    CreateTasksAction,
    FinishAction,
    OrchestratorDecision,
    RetryTaskAction,
    SplitTaskAction,
    ValidateTaskAction,
    WaitAction,
)
from evoweave_ds.orchestration.progress_detector import ProgressDetector, ProgressState
from evoweave_ds.orchestration.result_reducer import ResultReducer
from evoweave_ds.orchestration.scheduler import AgentAllocationDecision, Scheduler, TaskLease
from evoweave_ds.orchestration.task_graph import TaskGraph


class DecisionSource(Protocol):
    def decide(self, control_view: OrchestrationControlView) -> OrchestratorDecision: ...


@dataclass(frozen=True, slots=True)
class _RollbackState:
    snapshot: GraphSnapshot
    task_specs: tuple[TaskSpec, ...]
    leases: dict[SpecId, TaskLease]
    execution_specs: dict[SpecId, AgentExecutionSpec]
    allocation_decisions: list[AgentAllocationDecision]
    results: list[ResultControlSummary]
    processed_decisions: dict[SpecId, str]
    progress_state: ProgressState | None
    decision_count: int
    checkpoint_version: int
    acceptance_satisfied: bool
    finished: bool
    finish_summary: str | None


class Orchestrator:
    def __init__(
        self,
        *,
        graph: TaskGraph,
        graph_store: GraphStateStore,
        decision_ledger: DecisionLedger,
        checkpoint_manager: CheckpointManager,
        policy: GraphPolicy | None = None,
    ) -> None:
        self._graph = graph
        self._graph_store = graph_store
        self._decision_ledger = decision_ledger
        self._checkpoint_manager = checkpoint_manager
        self._policy = policy or GraphPolicy()
        self._execution_specs: dict[SpecId, AgentExecutionSpec] = {}
        self._leases: dict[SpecId, TaskLease] = {}
        self._results: list[ResultControlSummary] = []
        self._allocation_decisions: list[AgentAllocationDecision] = []
        self._processed_decisions: dict[SpecId, str] = {}
        self._progress_detector = ProgressDetector()
        self._progress_state: ProgressState | None = None
        self._decision_count = 0
        self._checkpoint_version = 0
        self._acceptance_satisfied = False
        self._finished = False
        self._finish_summary: str | None = None
        self._persist()

    @property
    def graph(self) -> TaskGraph:
        return self._graph

    @property
    def active_leases(self) -> tuple[TaskLease, ...]:
        return tuple(
            lease for lease in self._leases.values() if lease.status is TaskLeaseStatus.ACTIVE
        )

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def allocation_decisions(self) -> tuple[AgentAllocationDecision, ...]:
        return tuple(self._allocation_decisions)

    @property
    def finish_summary(self) -> str | None:
        return self._finish_summary

    def control_view(self) -> OrchestrationControlView:
        tasks = tuple(
            TaskControlItem(
                task_id=node.task_id,
                task_spec_version=node.task_spec_version,
                status=node.status,
                attempts=node.attempts,
                goal=self._graph.spec_for(node.task_id).goal[:2_000],
                write_scope=self._graph.spec_for(node.task_id).write_scope,
            )
            for node in sorted(self._graph.snapshot.nodes, key=lambda item: str(item.task_id))
        )
        return OrchestrationControlView(
            run_id=self._graph.snapshot.run_id,
            graph_version=self._graph.snapshot.version,
            tasks=tasks,
            recent_results=tuple(self._results[-20:]),
            acceptance_satisfied=self._acceptance_satisfied,
            decision_count=self._decision_count,
        )

    def step(self, source: DecisionSource) -> bool:
        return self.apply(source.decide(self.control_view()))

    def apply(self, decision: OrchestratorDecision) -> bool:
        payload = decision.model_dump_json().encode("utf-8")
        payload_digest = sha256(payload).hexdigest()
        processed_digest = self._processed_decisions.get(decision.decision_id)
        if processed_digest is not None:
            if processed_digest != payload_digest:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_ERROR,
                    "同一调度决策 ID 不能重放不同内容",
                )
            return False
        ledger_payload = self._decision_ledger.get_decision_payload(decision.decision_id)
        if ledger_payload is not None:
            if ledger_payload != payload:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_ERROR,
                    "决策账本中的 ID 已绑定不同内容",
                )
            return False
        if self._finished:
            raise DomainError(ErrorCode.INVALID_STATE_TRANSITION, "运行已经结束")
        if decision.run_id != self._graph.snapshot.run_id:
            raise DomainError(ErrorCode.INVALID_SPEC, "调度决策属于其他 run")
        if decision.based_on_graph_version != self._graph.snapshot.version:
            raise DomainError(ErrorCode.INVALID_STATE_TRANSITION, "调度决策基于过期图版本")
        if self._decision_count >= self._policy.max_decisions:
            raise DomainError(ErrorCode.POLICY_REJECTED, "调度决策次数达到上限")

        before = self._rollback_state()
        try:
            self._apply_action(decision)
            self._decision_count += 1
            self._processed_decisions[decision.decision_id] = payload_digest
            self._progress_state = self._progress_detector.observe(
                self._graph.snapshot,
                self._progress_state,
            )
            if self._progress_state.unchanged_decisions > self._policy.max_no_progress_decisions:
                raise DomainError(ErrorCode.POLICY_REJECTED, "调度连续无进展")
            self._persist()
        except Exception:
            self._restore_rollback_state(before)
            raise
        self._decision_ledger.record_decision(
            decision_id=decision.decision_id,
            run_id=decision.run_id,
            graph_version=decision.based_on_graph_version,
            payload=payload,
        )
        return True

    def dispatch(
        self,
        *,
        scheduler: Scheduler,
        agent_factory: AgentFactory,
        capability_plan_for: Callable[[TaskId], CapabilityPlan],
    ) -> tuple[AgentExecutionSpec, ...]:
        if self._finished:
            return ()
        before = self._rollback_state()
        try:
            allocation = scheduler.plan(self._graph, self.active_leases)
            self._allocation_decisions.append(allocation)
            selected = allocation.selected_task_ids
            execution_specs: list[AgentExecutionSpec] = []
            for task_id in selected:
                prior_versions = [
                    spec.version
                    for spec in self._execution_specs.values()
                    if spec.task_id == task_id
                ]
                execution_spec = agent_factory.create(
                    run_id=self._graph.snapshot.run_id,
                    task_spec=self._graph.spec_for(task_id),
                    capability_plan=capability_plan_for(task_id),
                    version=max(prior_versions, default=0) + 1,
                )
                lease = scheduler.lease(self._graph, execution_spec)
                self._graph.transition(task_id, TaskStatus.RUNNING)
                self._execution_specs[execution_spec.spec_id] = execution_spec
                self._leases[execution_spec.spec_id] = lease
                execution_specs.append(execution_spec)
            self._persist()
            return tuple(execution_specs)
        except Exception:
            self._restore_rollback_state(before)
            raise

    def accept_result(
        self,
        result: TaskResult,
        *,
        suggestions: tuple[TaskSuggestion, ...] = (),
    ) -> ResultControlSummary:
        try:
            execution_spec = self._execution_specs[result.execution_spec_id]
            lease = self._leases[result.execution_spec_id]
        except KeyError as exc:
            raise DomainError(ErrorCode.INVALID_SPEC, "结果没有已知的执行规格和租约") from exc
        before = self._rollback_state()
        try:
            summary = ResultReducer().reduce(
                result=result,
                execution_spec=execution_spec,
                suggestions=suggestions,
            )
            target_status = {
                ResultStatus.SUCCEEDED: TaskStatus.SUCCEEDED,
                ResultStatus.FAILED: TaskStatus.FAILED,
                ResultStatus.CANCELLED: TaskStatus.CANCELLED,
                ResultStatus.BLOCKED: TaskStatus.BLOCKED,
            }[result.status]
            self._graph.transition(result.task_id, target_status)
            self._leases[result.execution_spec_id] = lease.model_copy(
                update={"status": TaskLeaseStatus.COMPLETED}
            )
            self._results.append(summary)
            self._persist()
            return summary
        except Exception:
            self._restore_rollback_state(before)
            raise

    def reroute_execution(
        self,
        execution_spec_id: SpecId,
        decision: ModelRoutingDecision,
        *,
        scheduler: Scheduler,
    ) -> AgentExecutionSpec:
        """Replace a failed model invocation with a new immutable worker instance."""

        try:
            previous = self._execution_specs[execution_spec_id]
            previous_lease = self._leases[execution_spec_id]
        except KeyError as exc:
            raise DomainError(ErrorCode.INVALID_SPEC, "回退目标没有已知执行规格和租约") from exc
        if previous_lease.status is not TaskLeaseStatus.ACTIVE:
            raise DomainError(ErrorCode.INVALID_STATE_TRANSITION, "只能替换仍处于活跃租约的 Agent")

        before = self._rollback_state()
        try:
            revised = revise_execution_spec_for_routing(previous, decision)
            self._graph.transition(previous.task_id, TaskStatus.FAILED)
            self._graph.transition(previous.task_id, TaskStatus.READY)
            self._leases[execution_spec_id] = previous_lease.model_copy(
                update={"status": TaskLeaseStatus.RELEASED}
            )
            revised_lease = scheduler.lease(self._graph, revised)
            self._graph.transition(previous.task_id, TaskStatus.RUNNING)
            self._execution_specs[revised.spec_id] = revised
            self._leases[revised.spec_id] = revised_lease
            self._persist()
            return revised
        except Exception:
            self._restore_rollback_state(before)
            raise

    def continuation_spec(
        self,
        execution_spec_id: SpecId,
        *,
        scheduler: Scheduler,
    ) -> AgentExecutionSpec:
        """续接同一任务: 生成下一版本的 continuable 执行规格(带上下文重试)。

        借鉴 dsh 可续接子代理(followup/resume): 失败 Worker 不重开全新
        会话, 而是基于上一执行规格派生新版本, 由 WorkerRuntime 注入失败
        诊断后带上下文重试。
        """
        try:
            previous = self._execution_specs[execution_spec_id]
            previous_lease = self._leases[execution_spec_id]
        except KeyError as exc:
            raise DomainError(ErrorCode.INVALID_SPEC, "续接目标没有已知执行规格和租约") from exc
        if previous_lease.status is not TaskLeaseStatus.ACTIVE:
            raise DomainError(ErrorCode.INVALID_STATE_TRANSITION, "只能续接仍处于活跃租约的 Agent")

        before = self._rollback_state()
        try:
            continued = previous.model_copy(
                update={
                    "spec_id": SpecId.new(),
                    "agent_id": AgentId.new(),
                    "version": previous.version + 1,
                    "continuable": True,
                    "parent_spec_id": previous.spec_id,
                }
            )
            self._graph.transition(previous.task_id, TaskStatus.READY)
            self._leases[execution_spec_id] = previous_lease.model_copy(
                update={"status": TaskLeaseStatus.RELEASED}
            )
            continued_lease = scheduler.lease(self._graph, continued)
            self._graph.transition(previous.task_id, TaskStatus.RUNNING)
            self._execution_specs[continued.spec_id] = continued
            self._leases[continued.spec_id] = continued_lease
            self._persist()
            return continued
        except Exception:
            self._restore_rollback_state(before)
            raise

    def expire_leases(
        self,
        *,
        scheduler: Scheduler,
        now: datetime | None = None,
    ) -> tuple[TaskLease, ...]:
        """Expire overdue leases through the same recoverable control plane."""

        before = self._rollback_state()
        expired: list[TaskLease] = []
        try:
            for spec_id, lease in tuple(self._leases.items()):
                if lease.status is not TaskLeaseStatus.ACTIVE:
                    continue
                revised = scheduler.expire(self._graph, lease, now=now)
                if revised.status is TaskLeaseStatus.EXPIRED:
                    self._leases[spec_id] = revised
                    expired.append(revised)
            if expired:
                self._persist()
            return tuple(expired)
        except Exception:
            self._restore_rollback_state(before)
            raise

    def mark_acceptance_satisfied(self) -> None:
        before = self._rollback_state()
        try:
            self._acceptance_satisfied = True
            self._persist()
        except Exception:
            self._restore_rollback_state(before)
            raise

    def _apply_action(self, decision: OrchestratorDecision) -> None:
        action = decision.action
        if isinstance(action, CreateTasksAction):
            self._validate_new_tasks(action.task_specs)
            self._graph.add_tasks(action.task_specs)
        elif isinstance(action, SplitTaskAction):
            self._validate_new_tasks(action.task_specs)
            source = self._graph.node_for(action.source_task_id)
            if action.cancel_source and any(
                action.source_task_id in spec.depends_on for spec in action.task_specs
            ):
                raise DomainError(
                    ErrorCode.INVALID_SPEC,
                    "替代已取消源任务的子任务不能依赖该源任务",
                )
            if action.cancel_source and source.status not in {
                TaskStatus.SUCCEEDED,
                TaskStatus.CANCELLED,
            }:
                self._graph.transition(action.source_task_id, TaskStatus.CANCELLED)
            edges = tuple(
                TaskEdge(
                    source=action.source_task_id,
                    target=spec.task_id,
                    relation=(
                        TaskRelation.SUPERSEDES if action.cancel_source else TaskRelation.DEPENDS_ON
                    ),
                )
                for spec in action.task_specs
                if action.source_task_id not in spec.depends_on
            )
            self._graph.add_tasks(action.task_specs, extra_edges=edges)
        elif isinstance(action, CancelTaskAction):
            self._graph.transition(action.task_id, TaskStatus.CANCELLED)
            for spec_id, lease in tuple(self._leases.items()):
                if lease.task_id == action.task_id and lease.status is TaskLeaseStatus.ACTIVE:
                    self._leases[spec_id] = lease.model_copy(
                        update={"status": TaskLeaseStatus.RELEASED}
                    )
        elif isinstance(action, RetryTaskAction):
            node = self._graph.node_for(action.task_id)
            if node.status not in {TaskStatus.FAILED, TaskStatus.BLOCKED}:
                raise DomainError(ErrorCode.INVALID_STATE_TRANSITION, "只能重试失败或阻塞任务")
            if action.replacement_spec is not None:
                if action.replacement_spec.task_id != action.task_id:
                    raise DomainError(ErrorCode.INVALID_SPEC, "替换规格必须保持 task_id")
                self._graph.replace_spec(action.replacement_spec)
            self._graph.transition(action.task_id, TaskStatus.READY)
        elif isinstance(action, ValidateTaskAction):
            self._validate_new_tasks((action.validation_spec,))
            validation_edges: tuple[TaskEdge, ...] = (
                TaskEdge(
                    source=action.validated_task_id,
                    target=action.validation_spec.task_id,
                    relation=TaskRelation.VALIDATES,
                ),
            )
            if action.validated_task_id not in action.validation_spec.depends_on:
                validation_edges += (
                    TaskEdge(
                        source=action.validated_task_id,
                        target=action.validation_spec.task_id,
                        relation=TaskRelation.DEPENDS_ON,
                    ),
                )
            self._graph.add_tasks((action.validation_spec,), extra_edges=validation_edges)
        elif isinstance(action, WaitAction):
            return
        elif isinstance(action, FinishAction):
            if not self._acceptance_satisfied or self.active_leases:
                raise DomainError(ErrorCode.POLICY_REJECTED, "验收未满足或仍有活跃 Agent")
            unfinished = [
                node
                for node in self._graph.snapshot.nodes
                if node.status
                not in {TaskStatus.SUCCEEDED, TaskStatus.CANCELLED, TaskStatus.FAILED}
            ]
            if unfinished:
                raise DomainError(ErrorCode.POLICY_REJECTED, "仍有未终止任务，不能结束")
            self._finished = True
            self._finish_summary = action.summary
        else:
            raise DomainError(ErrorCode.INVALID_SPEC, "未知调度动作")

    def _validate_new_tasks(self, specs: tuple[TaskSpec, ...]) -> None:
        if len(specs) > self._policy.max_tasks_per_decision:
            raise DomainError(ErrorCode.POLICY_REJECTED, "单次决策创建任务过多")
        self._progress_detector.reject_duplicate_specs(specs, self._graph.task_specs)
        if any(spec.change_spec_id != self._graph.task_specs[0].change_spec_id for spec in specs):
            raise DomainError(ErrorCode.INVALID_SPEC, "新任务必须属于同一 ChangeSpec")

    def _persist(self) -> None:
        for snapshot, specs in self._graph.version_records:
            self._graph_store.save_graph(snapshot, specs)
        self._checkpoint_version += 1
        self._checkpoint_manager.save(self._checkpoint())

    def _checkpoint(self) -> OrchestrationCheckpoint:
        return OrchestrationCheckpoint(
            run_id=self._graph.snapshot.run_id,
            version=self._checkpoint_version,
            graph=self._graph.snapshot,
            task_specs=self._graph.task_specs,
            execution_specs=tuple(self._execution_specs.values()),
            active_leases=self.active_leases,
            allocation_decisions=tuple(self._allocation_decisions),
            result_summaries=tuple(self._results),
            processed_decisions=tuple(
                ProcessedDecisionRecord(decision_id=decision_id, payload_sha256=digest)
                for decision_id, digest in self._processed_decisions.items()
            ),
            progress_state=self._progress_state,
            decision_count=self._decision_count,
            acceptance_satisfied=self._acceptance_satisfied,
            finished=self._finished,
            finish_summary=self._finish_summary,
        )

    @classmethod
    def restore(
        cls,
        checkpoint: OrchestrationCheckpoint,
        *,
        graph_store: GraphStateStore,
        decision_ledger: DecisionLedger,
        checkpoint_manager: CheckpointManager,
        policy: GraphPolicy | None = None,
    ) -> "Orchestrator":
        instance = cls.__new__(cls)
        instance._graph = TaskGraph(
            snapshot=checkpoint.graph,
            task_specs=checkpoint.task_specs,
            policy=policy,
        )
        instance._graph_store = graph_store
        instance._decision_ledger = decision_ledger
        instance._checkpoint_manager = checkpoint_manager
        instance._policy = policy or GraphPolicy()
        instance._execution_specs = {spec.spec_id: spec for spec in checkpoint.execution_specs}
        active_lease_by_spec = {
            lease.execution_spec_id: lease for lease in checkpoint.active_leases
        }
        instance._leases = active_lease_by_spec
        instance._results = list(checkpoint.result_summaries)
        instance._allocation_decisions = list(checkpoint.allocation_decisions)
        instance._processed_decisions = {
            item.decision_id: item.payload_sha256 for item in checkpoint.processed_decisions
        }
        instance._progress_detector = ProgressDetector()
        instance._progress_state = checkpoint.progress_state
        instance._decision_count = checkpoint.decision_count
        instance._checkpoint_version = checkpoint.version
        instance._acceptance_satisfied = checkpoint.acceptance_satisfied
        instance._finished = checkpoint.finished
        instance._finish_summary = checkpoint.finish_summary
        return instance

    def _rollback_state(self) -> _RollbackState:
        return _RollbackState(
            snapshot=self._graph.snapshot,
            task_specs=self._graph.task_specs,
            leases=dict(self._leases),
            execution_specs=dict(self._execution_specs),
            allocation_decisions=list(self._allocation_decisions),
            results=list(self._results),
            processed_decisions=dict(self._processed_decisions),
            progress_state=self._progress_state,
            decision_count=self._decision_count,
            checkpoint_version=self._checkpoint_version,
            acceptance_satisfied=self._acceptance_satisfied,
            finished=self._finished,
            finish_summary=self._finish_summary,
        )

    def _restore_rollback_state(self, state: _RollbackState) -> None:
        self._graph = TaskGraph(
            snapshot=state.snapshot,
            task_specs=state.task_specs,
            policy=self._policy,
        )
        self._leases = state.leases
        self._execution_specs = state.execution_specs
        self._allocation_decisions = state.allocation_decisions
        self._results = state.results
        self._processed_decisions = state.processed_decisions
        self._progress_state = state.progress_state
        self._decision_count = state.decision_count
        self._checkpoint_version = state.checkpoint_version
        self._acceptance_satisfied = state.acceptance_satisfied
        self._finished = state.finished
        self._finish_summary = state.finish_summary


class ScriptedDecisionSource:
    def __init__(self, decisions: tuple[OrchestratorDecision, ...]) -> None:
        self._decisions = list(decisions)

    def decide(self, control_view: OrchestrationControlView) -> OrchestratorDecision:
        if not self._decisions:
            raise DomainError(ErrorCode.SCRIPT_EXHAUSTED, "调度决策脚本已耗尽")
        decision = self._decisions.pop(0)
        if decision.run_id != control_view.run_id:
            raise DomainError(ErrorCode.INVALID_SPEC, "脚本决策属于其他 run")
        return decision
