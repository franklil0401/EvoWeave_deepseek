"""Single-task end-to-end update path used by CLI and deterministic E2E tests."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from evoweave_ds.agent_runtime.context_builder import ContextBuilder
from evoweave_ds.agent_runtime.runtime import WorkerRuntime
from evoweave_ds.application.adaptive_task_planner import (
    AdaptiveTaskPlan,
    AdaptiveTaskPlanner,
    TaskPlanner,
)
from evoweave_ds.application.configuration import EvoWeaveConfig
from evoweave_ds.application.run_state import JsonRunStateStore
from evoweave_ds.application.runtime_layout import RuntimeLayout
from evoweave_ds.capabilities.builtins import default_capabilities
from evoweave_ds.capabilities.registry import CapabilityRegistry
from evoweave_ds.capabilities.tool_executor import ToolExecutor
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.artifacts import PatchArtifact
from evoweave_ds.domain.enums import (
    IntegrationStatus,
    ResultStatus,
    RiskLevel,
    RunStatus,
    WorkspaceAccessMode,
    WorkspaceLeaseStatus,
)
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, SpecId, WorkspaceId
from evoweave_ds.domain.integration_models import (
    IntegratedPatchSet,
    IntegrationWorkspaceState,
    ValidationCommand,
    ValidationReport,
)
from evoweave_ds.domain.model_routing import (
    ModelProfile,
    ModelRoutingDecision,
)
from evoweave_ds.domain.policies import GraphPolicy
from evoweave_ds.domain.ports import ArtifactStore, CommandRunner, ModelGateway, ModelRouter
from evoweave_ds.domain.repository_models import RepositoryProfile
from evoweave_ds.domain.resources import RuntimeLimits
from evoweave_ds.domain.run_models import RunManifest
from evoweave_ds.domain.task_result import TaskResult
from evoweave_ds.domain.workspace_models import WorkspaceLease
from evoweave_ds.infrastructure.models.fixed_router import FixedReasoningRouter
from evoweave_ds.infrastructure.persistence.graph_repository import SQLiteOrchestrationStore
from evoweave_ds.infrastructure.persistence.sqlite import SQLiteDatabase
from evoweave_ds.infrastructure.telemetry.jsonl import JsonlEventRecorder
from evoweave_ds.integration.final_diff import FinalDiffExporter
from evoweave_ds.integration.integration_workspace import IntegrationWorkspaceManager
from evoweave_ds.integration.patch_applier import PatchApplier
from evoweave_ds.integration.patch_guard import PatchGuard
from evoweave_ds.integration.service import PatchIntegrationService
from evoweave_ds.integration.state_store import JsonIntegrationStateStore
from evoweave_ds.orchestration.agent_factory import AgentFactory, CapabilityPlan
from evoweave_ds.orchestration.checkpointing import CheckpointManager
from evoweave_ds.orchestration.decisions import (
    FinishAction,
    OrchestratorDecision,
)
from evoweave_ds.orchestration.orchestrator import Orchestrator
from evoweave_ds.orchestration.scheduler import Scheduler
from evoweave_ds.orchestration.task_graph import TaskGraph
from evoweave_ds.validation.gate import DeterministicValidationGate
from evoweave_ds.validation.plan import PythonValidationPlanBuilder
from evoweave_ds.workspaces.manager import WorkspaceManager
from evoweave_ds.workspaces.patch_collector import GitPatchCollector
from evoweave_ds.workspaces.state_store import JsonWorkspaceLeaseStore


@dataclass(frozen=True, slots=True)
class UpdateWorkflowOutcome:
    manifest: RunManifest
    task_results: tuple[TaskResult, ...]
    final_patch: IntegratedPatchSet
    validation_report: ValidationReport

    @property
    def task_result(self) -> TaskResult:
        return self.task_results[0]

    @property
    def agent_count(self) -> int:
        return len(self.task_results)


ValidationRunnerFactory = Callable[[WorkspaceLease], CommandRunner]


def prepare_task_plan(
    *,
    config: EvoWeaveConfig,
    run_store: JsonRunStateStore,
    manifest: RunManifest,
    profile: RepositoryProfile,
    approve_high_risk: bool,
    task_planner: TaskPlanner | None = None,
) -> AdaptiveTaskPlan:
    plan = (task_planner or AdaptiveTaskPlanner(config)).plan(manifest, profile)
    high_risk_tasks = tuple(item for item in plan.task_specs if item.risk_level is RiskLevel.HIGH)
    if high_risk_tasks and not approve_high_risk:
        if manifest.status is not RunStatus.WAITING_FOR_INPUT:
            run_store.transition(
                manifest.run_id,
                RunStatus.WAITING_FOR_INPUT,
                message=f"{len(high_risk_tasks)} 个高风险任务等待人工批准",
            )
        raise DomainError(
            ErrorCode.APPROVAL_REQUIRED,
            "检测到高风险更新；请审查任务范围后显式批准",
        )
    return plan


class SingleTaskUpdateWorkflow:
    def __init__(
        self,
        *,
        config: EvoWeaveConfig,
        layout: RuntimeLayout,
        run_store: JsonRunStateStore,
        artifact_store: ArtifactStore,
        model_gateway: ModelGateway,
        model_profiles: tuple[ModelProfile, ...],
        validation_runner_factory: ValidationRunnerFactory,
        approve_high_risk: bool = False,
        task_planner: TaskPlanner | None = None,
        model_router: ModelRouter | None = None,
        additional_validation_commands: tuple[ValidationCommand, ...] = (),
    ) -> None:
        self._config = config
        self._layout = layout
        self._run_store = run_store
        self._artifact_store = artifact_store
        self._model_gateway = model_gateway
        self._model_profiles = model_profiles
        self._validation_runner_factory = validation_runner_factory
        self._approve_high_risk = approve_high_risk
        self._task_planner = task_planner
        self._model_router = model_router
        self._additional_validation_commands = additional_validation_commands

    def execute(
        self,
        manifest: RunManifest,
        profile: RepositoryProfile,
    ) -> UpdateWorkflowOutcome:
        if manifest.status not in {RunStatus.ANALYZED, RunStatus.WAITING_FOR_INPUT}:
            raise DomainError(
                ErrorCode.INVALID_STATE_TRANSITION,
                "只能执行 analyzed 或 waiting_for_input 运行",
            )
        if not manifest.change_spec.allowed_paths:
            raise DomainError(ErrorCode.INVALID_SPEC, "执行更新前必须显式提供至少一个允许修改路径")
        task_plan = prepare_task_plan(
            config=self._config,
            run_store=self._run_store,
            manifest=manifest,
            profile=profile,
            approve_high_risk=self._approve_high_risk,
            task_planner=self._task_planner,
        )
        manifest = self._run_store.transition(
            manifest.run_id,
            RunStatus.RUNNING,
            message=f"动态更新流水线开始执行：{task_plan.rationale}",
        )
        worker_manager = WorkspaceManager(
            repository_root=manifest.change_spec.repository,
            worktree_root=self._layout.worker_worktrees,
            lease_store=JsonWorkspaceLeaseStore(self._layout.worker_state),
        )
        integration_store = JsonIntegrationStateStore(self._layout.integration_state)
        integration_manager = IntegrationWorkspaceManager(
            repository_root=manifest.change_spec.repository,
            worktree_root=self._layout.integration_worktrees,
            state_store=integration_store,
        )
        baseline_manager = WorkspaceManager(
            repository_root=manifest.change_spec.repository,
            worktree_root=self._layout.baseline_worktrees,
            lease_store=JsonWorkspaceLeaseStore(self._layout.baseline_state),
        )
        worker_leases: list[WorkspaceLease] = []
        baseline_lease = None
        try:
            orchestration_store = SQLiteOrchestrationStore(
                SQLiteDatabase(self._layout.orchestration_database)
            )
            graph_policy = GraphPolicy()
            orchestrator = Orchestrator(
                graph=TaskGraph.create(
                    run_id=manifest.run_id,
                    task_specs=task_plan.task_specs,
                    policy=graph_policy,
                ),
                graph_store=orchestration_store,
                decision_ledger=orchestration_store,
                checkpoint_manager=CheckpointManager(orchestration_store),
                policy=graph_policy,
            )
            scheduler = Scheduler(graph_policy)
            agent_factory = AgentFactory(
                model_router=self._model_router or FixedReasoningRouter(),
                model_profiles=self._model_profiles,
            )
            task_spec_by_id = {item.task_id: item for item in task_plan.task_specs}
            runtime = WorkerRuntime(
                model_gateway=self._model_gateway,
                tool_executor=ToolExecutor(CapabilityRegistry(default_capabilities())),
                context_builder=ContextBuilder(self._artifact_store),
                artifact_store=self._artifact_store,
                workspace_provider=worker_manager,
                event_recorder=JsonlEventRecorder(self._layout.events),
            )
            successful_executions: list[AgentExecutionSpec] = []
            task_results: list[TaskResult] = []
            patches: list[PatchArtifact] = []
            completed_tasks = 0
            while completed_tasks < len(task_plan.task_specs):
                batch = orchestrator.dispatch(
                    scheduler=scheduler,
                    agent_factory=agent_factory,
                    capability_plan_for=lambda task_id: CapabilityPlan(
                        tool_names=(
                            ("file.read", "file.search", "file.write")
                            if task_spec_by_id[task_id].write_scope
                            else ("file.read", "file.search")
                        ),
                        runtime_limits=RuntimeLimits(
                            max_steps=self._config.max_worker_steps,
                            max_tool_calls=self._config.max_worker_tool_calls,
                            timeout_seconds=self._config.max_worker_seconds,
                        ),
                    ),
                )
                if not batch:
                    raise DomainError(ErrorCode.INVALID_GRAPH, "任务图没有可执行任务且尚未完成")
                initial_leases: list[WorkspaceLease] = []
                for execution in batch:
                    lease = worker_manager.create(execution)
                    worker_leases.append(lease)
                    initial_leases.append(lease)
                initial_results: tuple[TaskResult, ...]
                if len(batch) == 1:
                    initial_results = (runtime.execute(batch[0]),)
                else:
                    with ThreadPoolExecutor(max_workers=len(batch)) as executor_pool:
                        initial_results = tuple(executor_pool.map(runtime.execute, batch))
                for execution, initial_lease, initial_result in zip(
                    batch,
                    initial_leases,
                    initial_results,
                    strict=True,
                ):
                    current_execution = execution
                    current_worker_lease = initial_lease
                    result = initial_result
                    while True:
                        task_results.append(result)
                        fallback = _fallback_decision(
                            current_execution,
                            result,
                            self._model_profiles,
                            max_attempts=graph_policy.max_attempts_per_task,
                        )
                        if fallback is not None:
                            current_execution = orchestrator.reroute_execution(
                                current_execution.spec_id,
                                fallback,
                                scheduler=scheduler,
                            )
                            current_worker_lease = worker_manager.create(current_execution)
                            worker_leases.append(current_worker_lease)
                            result = runtime.execute(current_execution)
                            continue
                        if _continuation_decision(
                            current_execution,
                            result,
                            max_attempts=graph_policy.max_attempts_per_task,
                        ):
                            continued = orchestrator.continuation_spec(
                                current_execution.spec_id,
                                scheduler=scheduler,
                            )
                            current_execution = continued
                            current_worker_lease = worker_manager.create(current_execution)
                            worker_leases.append(current_worker_lease)
                            result = runtime.execute(
                                current_execution,
                                resume_from=result,
                            )
                            continue
                        break
                    orchestrator.accept_result(result)
                    if result.status is not ResultStatus.SUCCEEDED:
                        failure = result.failure
                        raise DomainError(
                            failure.code if failure is not None else ErrorCode.INVALID_MODEL_OUTPUT,
                            failure.message if failure is not None else "Worker 未成功完成",
                            details=failure.details if failure is not None else None,
                        )
                    successful_executions.append(current_execution)
                    if current_execution.write_scope:
                        patches.append(
                            GitPatchCollector(self._artifact_store).collect(
                                lease=current_worker_lease,
                                execution_spec=current_execution,
                                supporting_artifacts=tuple(
                                    artifact
                                    for artifact in result.artifacts
                                    if artifact.kind.value in {"command_log", "test_report"}
                                ),
                            )
                        )
                    completed_tasks += 1
            applier = PatchApplier(integration_manager.worktree_root)
            integration_state = PatchIntegrationService(
                manager=integration_manager,
                guard=PatchGuard(self._artifact_store),
                applier=applier,
            ).integrate(
                run_id=manifest.run_id,
                base_commit=manifest.change_spec.base_commit,
                patches=tuple(patches),
                task_specs=task_plan.task_specs,
            )
            final_patch = FinalDiffExporter(
                self._artifact_store,
                integration_manager.worktree_root,
            ).export(integration_state)

            validation_execution = successful_executions[0]
            baseline_execution = validation_execution.model_copy(
                update={
                    "spec_id": SpecId.new(),
                    "agent_id": AgentId.new(),
                    "tool_names": (),
                    "write_scope": (),
                }
            )
            baseline_lease = baseline_manager.create(baseline_execution)
            candidate_lease = _candidate_validation_lease(
                integration_state,
                validation_execution,
            )
            test_paths = tuple(
                item.path
                for item in profile.files
                if item.kind == "test"
                and item.line_count > 0
                and item.path.casefold().endswith(".py")
                and "__pycache__" not in item.path.casefold().split("/")
                and item.path.casefold().startswith("tests/")
            )
            commands = (
                *PythonValidationPlanBuilder().build(
                    local_test_paths=test_paths[:8],
                    impacted_test_paths=test_paths,
                ),
                *self._additional_validation_commands,
            )
            validation_report = DeterministicValidationGate(self._artifact_store).run(
                state=integration_state,
                commands=commands,
                baseline_runner=self._validation_runner_factory(baseline_lease),
                candidate_runner=self._validation_runner_factory(candidate_lease),
            )
            if validation_report.accepted:
                orchestrator.mark_acceptance_satisfied()
                orchestrator.apply(
                    OrchestratorDecision(
                        decision_id=SpecId.new(),
                        run_id=manifest.run_id,
                        based_on_graph_version=orchestrator.graph.snapshot.version,
                        action=FinishAction(summary="补丁集成与确定性验证通过"),
                        rationale="所有验收门禁通过，结束当前运行",
                    )
                )
            final_status = RunStatus.COMPLETED if validation_report.accepted else RunStatus.FAILED
            manifest = self._run_store.transition(
                manifest.run_id,
                final_status,
                message=(
                    "补丁集成与确定性验证通过"
                    if validation_report.accepted
                    else "候选补丁未通过确定性验证"
                ),
                error_code=(None if validation_report.accepted else ErrorCode.VALIDATION_FAILED),
                final_patch_artifact_id=final_patch.ref.artifact_id,
                validation_report_artifact_id=validation_report.report_ref.artifact_id
                if validation_report.report_ref is not None
                else None,
            )
            return UpdateWorkflowOutcome(
                manifest=manifest,
                task_results=tuple(task_results),
                final_patch=final_patch,
                validation_report=validation_report,
            )
        except DomainError as exc:
            current = self._run_store.get(manifest.run_id)
            if current.status is RunStatus.RUNNING:
                self._run_store.transition(
                    manifest.run_id,
                    RunStatus.FAILED,
                    message=exc.message,
                    error_code=exc.code,
                )
            raise
        finally:
            for worker_lease in worker_leases:
                worker_manager.release(worker_lease.workspace_id)
            if baseline_lease is not None:
                baseline_manager.release(baseline_lease.workspace_id)
            for state in integration_store.list_all():
                if state.run_id == manifest.run_id and state.status in {
                    IntegrationStatus.ACTIVE,
                    IntegrationStatus.FAILED,
                }:
                    integration_manager.release(state.integration_id)


def _fallback_decision(
    execution: AgentExecutionSpec,
    result: TaskResult,
    profiles: tuple[ModelProfile, ...],
    *,
    max_attempts: int,
) -> ModelRoutingDecision | None:
    """本阶段单一模型策略: 无回退链, 任何失败直接结构化失败."""
    return None


def _continuation_decision(
    execution: AgentExecutionSpec,
    result: TaskResult,
    *,
    max_attempts: int,
) -> bool:
    """续接重试决策(借鉴 dsh 可续接子代理)。

    仅当执行规格标记 continuable 且任务图允许重试时, 对失败 Worker 做
    带上下文的续接重试(复用同一任务, 注入失败诊断); 其余情况保持既有
    行为(直接结构化失败)。
    """
    if not execution.continuable:
        return False
    if execution.version >= max_attempts:
        return False
    if result.failure is None or not result.failure.retryable:
        # 失败必须可重试(如 MODEL_UNAVAILABLE / 验收未通过), 否则不续接。
        return result.failure is not None and result.status.value != "succeeded"
    return True


def _candidate_validation_lease(
    state: IntegrationWorkspaceState,
    execution: AgentExecutionSpec,
) -> WorkspaceLease:
    return WorkspaceLease(
        workspace_id=WorkspaceId.new(),
        run_id=execution.run_id,
        task_id=execution.task_id,
        agent_id=execution.agent_id,
        execution_spec_id=execution.spec_id,
        execution_spec_version=execution.version,
        repository_root=state.repository_root,
        worktree_path=state.worktree_path,
        branch_name=state.branch_name,
        base_commit=state.base_commit,
        access_mode=WorkspaceAccessMode.READ_ONLY,
        read_scope=execution.read_scope,
        status=WorkspaceLeaseStatus.ACTIVE,
    )
