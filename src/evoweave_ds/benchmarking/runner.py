"""Execute one locked benchmark combination and persist an honest run record."""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic

from evoweave_ds.agent_runtime.context_builder import ContextBuilder
from evoweave_ds.application.adaptive_task_planner import AdaptiveTaskPlan
from evoweave_ds.application.analysis_service import AnalysisService
from evoweave_ds.application.configuration import EvoWeaveConfig
from evoweave_ds.application.intake_service import IntakeService
from evoweave_ds.application.run_state import JsonRunStateStore
from evoweave_ds.application.runtime_layout import RuntimeLayout
from evoweave_ds.application.update_workflow import (
    SingleTaskUpdateWorkflow,
    UpdateWorkflowOutcome,
)
from evoweave_ds.benchmarking.materializer import FixtureMaterializer
from evoweave_ds.benchmarking.models import (
    AgentStrategy,
    BenchmarkRunRecord,
    BenchmarkRunStatus,
    BenchmarkSuite,
    BenchmarkTask,
    EvidenceLevel,
    ModelStrategy,
)
from evoweave_ds.benchmarking.strategies import (
    planner_for_strategy,
    profiles_for_strategy,
    router_for_strategy,
)
from evoweave_ds.domain.enums import (
    EventType,
    FailureClassification,
    ResultStatus,
    TaskDifficulty,
    TaskStatus,
    ValidationPhase,
    ValidationScope,
)
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import SpecId, TaskId
from evoweave_ds.domain.integration_models import ValidationCommand, ValidationReport
from evoweave_ds.domain.model_routing import ModelProfile
from evoweave_ds.domain.ports import ModelGateway, ModelRequest, ModelResponse
from evoweave_ds.domain.repository_models import RepositoryProfile, difficulty_rank
from evoweave_ds.domain.run_models import RunManifest
from evoweave_ds.domain.task_result import TaskResult
from evoweave_ds.infrastructure.artifacts.local_store import LocalArtifactStore
from evoweave_ds.infrastructure.persistence.graph_repository import SQLiteOrchestrationStore
from evoweave_ds.infrastructure.persistence.sqlite import SQLiteDatabase
from evoweave_ds.infrastructure.telemetry.jsonl import JsonlEventRecorder
from evoweave_ds.orchestration.checkpointing import CheckpointManager, OrchestrationCheckpoint
from evoweave_ds.workspaces.command_policy import LocalWorkspaceCommandRunner


@dataclass(frozen=True, slots=True)
class _FrozenPlanner:
    plan_value: AdaptiveTaskPlan

    def plan(
        self,
        manifest: RunManifest,
        profile: RepositoryProfile,
    ) -> AdaptiveTaskPlan:
        del manifest, profile
        return self.plan_value


class _OneShotUnavailableGateway:
    """Inject one auditable primary failure for the locked fallback scenario."""

    def __init__(self, delegate: ModelGateway) -> None:
        self._delegate = delegate
        self._injected = False

    def list_profiles(self) -> tuple[ModelProfile, ...]:
        return self._delegate.list_profiles()

    def complete(self, request: ModelRequest) -> ModelResponse:
        if not self._injected:
            self._injected = True
            raise DomainError(
                ErrorCode.MODEL_UNAVAILABLE,
                "benchmark 注入首选模型不可用故障",
                details={
                    "fault_injected": True,
                    "model_key": request.model_key,
                    "scenario": "model_fallback",
                },
            )
        return self._delegate.complete(request)


class BenchmarkRunner:
    def __init__(
        self,
        *,
        project_root: Path | str,
        model_gateway: ModelGateway,
        model_profiles: tuple[ModelProfile, ...],
        suite_sha256: str,
        config: EvoWeaveConfig | None = None,
        hidden_acceptance_source: str = ("benchmarks/任务集/隐藏验收/hidden_acceptance.py"),
        hidden_acceptance_sha256: str | None = None,
        evidence_output_root: Path | str | None = None,
        system_commit: str | None = None,
    ) -> None:
        self._project_root = Path(project_root).resolve(strict=True)
        self._gateway = model_gateway
        self._profiles = model_profiles
        if len(suite_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in suite_sha256
        ):
            raise ValueError("benchmark suite_sha256 无效")
        self._suite_sha256 = suite_sha256
        self._system_commit = system_commit or _resolve_system_commit(self._project_root)
        if len(self._system_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self._system_commit
        ):
            raise ValueError("benchmark system_commit 无效")
        self._config = config or EvoWeaveConfig(runtime_directory=".runtime-benchmark")
        self._materializer = FixtureMaterializer(self._project_root)
        hidden_root = (self._project_root / "benchmarks/任务集/隐藏验收").resolve(strict=True)
        self._hidden_acceptance = (self._project_root / hidden_acceptance_source).resolve(
            strict=True
        )
        if self._hidden_acceptance.parent != hidden_root:
            raise ValueError("benchmark 隐藏验收路径越界")
        if hidden_acceptance_sha256 is not None:
            actual = sha256(self._hidden_acceptance.read_bytes()).hexdigest()
            if actual != hidden_acceptance_sha256:
                raise ValueError("benchmark 隐藏验收摘要漂移")
        self._evidence_output_root = (
            Path(evidence_output_root).resolve() if evidence_output_root is not None else None
        )

    @property
    def system_commit(self) -> str:
        return self._system_commit

    def run(
        self,
        *,
        task: BenchmarkTask,
        agent_strategy: AgentStrategy,
        model_strategy: ModelStrategy,
        evidence_level: EvidenceLevel,
        trial_index: int = 1,
    ) -> BenchmarkRunRecord:
        if not 1 <= trial_index <= 1_000:
            raise ValueError("benchmark trial_index 必须在 1 到 1000 之间")
        started = monotonic()
        with tempfile.TemporaryDirectory(prefix=f"evoweave_ds-{task.benchmark_id}-") as temporary:
            temporary_root = Path(temporary)
            repository = self._materializer.materialize(
                task,
                temporary_root / "repository",
            ).path
            layout = RuntimeLayout.create(repository, self._config)
            artifact_store = LocalArtifactStore(layout.artifacts)
            run_store = JsonRunStateStore(layout.run_state)
            change = IntakeService().create(
                repository=repository,
                objective=task.objective,
                acceptance_criteria=task.acceptance_criteria,
                allowed_paths=task.allowed_paths,
                forbidden_paths=task.forbidden_paths,
            )
            manifest, profile = AnalysisService(
                run_store=run_store,
                artifact_store=artifact_store,
            ).analyze(change)
            planner = planner_for_strategy(agent_strategy, self._config)
            plan = planner.plan(manifest, profile)
            predicted_difficulty = max(
                (item.difficulty.difficulty for item in plan.task_specs),
                key=difficulty_rank,
            )
            selected_profiles = profiles_for_strategy(model_strategy, self._profiles)
            target_command = ValidationCommand(
                command_id=SpecId.new(),
                name=f"隐藏验收:{task.benchmark_id}",
                argv=("python", str(self._hidden_acceptance), task.benchmark_id),
                scope=ValidationScope.FULL,
                timeout_seconds=120,
            )
            try:
                task_gateway = _gateway_for_task(task, self._gateway)
                outcome = SingleTaskUpdateWorkflow(
                    config=self._config,
                    layout=layout,
                    run_store=run_store,
                    artifact_store=artifact_store,
                    model_gateway=task_gateway,
                    model_profiles=selected_profiles,
                    validation_runner_factory=lambda lease: LocalWorkspaceCommandRunner(
                        lease=lease,
                        allowed_commands=("python",),
                        allow_host_execution=True,
                    ),
                    approve_high_risk=True,
                    task_planner=_FrozenPlanner(plan),
                    model_router=router_for_strategy(model_strategy),
                    additional_validation_commands=(target_command,),
                ).execute(manifest, profile)
            except DomainError as exc:
                checkpoint = _load_checkpoint(layout, manifest)
                usage = _usage_from_events(layout, manifest)
                evidence_directory = self._persist_failure_evidence(
                    layout=layout,
                    manifest=manifest,
                    checkpoint=checkpoint,
                    code=exc.code,
                    message=exc.message,
                    details=exc.details,
                )
                return _failure_record(
                    task=task,
                    manifest=manifest,
                    plan=plan,
                    agent_strategy=agent_strategy,
                    model_strategy=model_strategy,
                    evidence_level=evidence_level,
                    predicted_difficulty=predicted_difficulty,
                    duration_ms=_elapsed_ms(started),
                    checkpoint=checkpoint,
                    artifact_store=artifact_store,
                    code=exc.code,
                    message=exc.message,
                    suite_sha256=self._suite_sha256,
                    system_commit=self._system_commit,
                    usage=usage,
                    evidence_directory=evidence_directory,
                    model_profiles=self._profiles,
                    trial_index=trial_index,
                )

            checkpoint = _load_checkpoint(layout, outcome.manifest)
            if checkpoint is None:
                raise ValueError("benchmark 完成后缺少总调度检查点")
            target_passed = _target_passed(
                outcome.validation_report,
                target_command.command_id,
            )
            regression_passed = _regression_passed(
                outcome.validation_report,
                target_command.command_id,
            )
            route_valid, fallback_count = _route_metrics(checkpoint, outcome.task_results)
            route_hard_constraints_satisfied = _route_hard_constraints_satisfied(
                checkpoint,
                self._profiles,
            )
            context = _context_metrics(checkpoint, artifact_store)
            system_accepted, system_violations = _system_acceptance(
                task,
                checkpoint,
                fallback_count=fallback_count,
            )
            evidence_directory = self._persist_success_evidence(
                layout=layout,
                outcome=outcome,
                checkpoint=checkpoint,
                artifact_store=artifact_store,
            )
            passed = target_passed and regression_passed and system_accepted
            return BenchmarkRunRecord(
                benchmark_id=task.benchmark_id,
                run_id=str(outcome.manifest.run_id),
                trial_index=trial_index,
                suite_sha256=self._suite_sha256,
                system_commit=self._system_commit,
                agent_strategy=agent_strategy,
                model_strategy=model_strategy,
                evidence_level=evidence_level,
                status=(BenchmarkRunStatus.PASSED if passed else BenchmarkRunStatus.FAILED),
                target_tests_passed=target_passed,
                full_regression_passed=regression_passed,
                localization_candidates=outcome.final_patch.changed_paths,
                patch_generated=True,
                patch_applied=True,
                patch_authorized=True,
                input_tokens=sum(item.usage.input_tokens for item in outcome.task_results),
                output_tokens=sum(item.usage.output_tokens for item in outcome.task_results),
                reasoning_tokens=sum(item.usage.reasoning_tokens for item in outcome.task_results),
                duration_ms=_elapsed_ms(started),
                agent_count=len(checkpoint.execution_specs),
                task_count=len(checkpoint.task_specs),
                invalid_task_count=_invalid_task_count(checkpoint),
                conflict_count=0,
                orchestrator_context_chars=context[0],
                worker_context_chars=context[1],
                initial_route_valid=route_valid,
                route_hard_constraints_satisfied=route_hard_constraints_satisfied,
                fallback_count=fallback_count,
                predicted_difficulty=predicted_difficulty,
                selected_model_keys=tuple(
                    dict.fromkeys(
                        item.model_routing.selected_model_key for item in checkpoint.execution_specs
                    )
                ),
                evidence_directory=evidence_directory,
                failure_reason=_completed_failure_reason(
                    passed=passed,
                    target_passed=target_passed,
                    regression_passed=regression_passed,
                    system_violations=system_violations,
                ),
            )

    def _persist_success_evidence(
        self,
        *,
        layout: RuntimeLayout,
        outcome: UpdateWorkflowOutcome,
        checkpoint: OrchestrationCheckpoint,
        artifact_store: LocalArtifactStore,
    ) -> str | None:
        if self._evidence_output_root is None:
            return None
        root = self._evidence_directory(str(outcome.manifest.run_id))
        _atomic_bytes(
            root / "最终补丁.diff",
            artifact_store.get_bytes(outcome.final_patch.ref.artifact_id),
        )
        _atomic_text(
            root / "验证报告.json",
            outcome.validation_report.model_dump_json(indent=2),
        )
        _atomic_text(root / "总调度检查点.json", checkpoint.model_dump_json(indent=2))
        _atomic_text(
            root / "任务结果.json",
            json.dumps(
                [item.model_dump(mode="json") for item in outcome.task_results],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )
        log_root = root / "命令日志"
        for observation in outcome.validation_report.observations:
            _atomic_bytes(
                log_root / f"{observation.log_ref.artifact_id}.json",
                artifact_store.get_bytes(observation.log_ref.artifact_id),
            )
        self._persist_events(layout, str(outcome.manifest.run_id), root)
        return self._portable_evidence_path(root)

    def _persist_failure_evidence(
        self,
        *,
        layout: RuntimeLayout,
        manifest: RunManifest,
        checkpoint: OrchestrationCheckpoint | None,
        code: ErrorCode,
        message: str,
        details: dict[str, object],
    ) -> str | None:
        if self._evidence_output_root is None:
            return None
        root = self._evidence_directory(str(manifest.run_id))
        _atomic_text(
            root / "失败信息.json",
            json.dumps(
                {"error_code": code.value, "message": message, "details": details},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )
        if checkpoint is not None:
            _atomic_text(root / "总调度检查点.json", checkpoint.model_dump_json(indent=2))
        self._persist_events(layout, str(manifest.run_id), root)
        return self._portable_evidence_path(root)

    def _evidence_directory(self, run_id: str) -> Path:
        assert self._evidence_output_root is not None
        root = (self._evidence_output_root / run_id).resolve()
        if root.parent != self._evidence_output_root:
            raise ValueError("benchmark 证据目录越界")
        if root.exists():
            raise ValueError("benchmark 运行证据目录已经存在")
        root.mkdir(parents=True)
        return root

    @staticmethod
    def _persist_events(layout: RuntimeLayout, run_id: str, root: Path) -> None:
        source = layout.events / f"{run_id}.jsonl"
        if source.exists():
            _atomic_bytes(root / "事件.jsonl", source.read_bytes())

    def _portable_evidence_path(self, root: Path) -> str:
        if root.is_relative_to(self._project_root):
            return root.relative_to(self._project_root).as_posix()
        return str(root)


class BenchmarkResultStore:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).resolve()

    def load(self) -> tuple[BenchmarkRunRecord, ...]:
        if not self._path.exists():
            return ()
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("benchmark 结果文件顶层必须是数组")
        return tuple(BenchmarkRunRecord.model_validate(item) for item in payload)

    def append(self, record: BenchmarkRunRecord) -> None:
        records = self.load()
        if any(item.suite_sha256 != record.suite_sha256 for item in records):
            raise ValueError("结果文件包含其他任务集版本")
        if any(item.system_commit != record.system_commit for item in records):
            raise ValueError("结果文件包含其他系统 Git 提交")
        key = _record_key(record)
        if any(_record_key(item) == key for item in records):
            raise ValueError("同一任务、策略、重复序号和真实性等级的 benchmark 记录已存在")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    [item.model_dump(mode="json") for item in (*records, record)],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self._path)
        finally:
            if temporary.exists():
                temporary.unlink()


def selected_tasks(
    suite: BenchmarkSuite,
    benchmark_ids: tuple[str, ...],
) -> tuple[BenchmarkTask, ...]:
    if not benchmark_ids:
        return suite.tasks
    task_by_id = {task.benchmark_id: task for task in suite.tasks}
    unknown = set(benchmark_ids).difference(task_by_id)
    if unknown:
        raise ValueError(f"未知 benchmark 任务：{sorted(unknown)}")
    if len(set(benchmark_ids)) != len(benchmark_ids):
        raise ValueError("benchmark 任务不能重复")
    return tuple(task_by_id[item] for item in benchmark_ids)


def _load_checkpoint(
    layout: RuntimeLayout,
    manifest: RunManifest,
) -> OrchestrationCheckpoint | None:
    return CheckpointManager(
        SQLiteOrchestrationStore(SQLiteDatabase(layout.orchestration_database))
    ).load(manifest.run_id)


def _target_passed(report: ValidationReport, command_id: SpecId) -> bool:
    candidate = tuple(
        item
        for item in report.observations
        if item.command_id == command_id and item.phase is ValidationPhase.CANDIDATE
    )
    return len(candidate) == 1 and candidate[0].exit_code == 0 and not candidate[0].timed_out


def _regression_passed(report: ValidationReport, target_command_id: SpecId) -> bool:
    blocking = {FailureClassification.NEW, FailureClassification.UNSTABLE}
    return not any(
        item.command_id != target_command_id and item.classification in blocking
        for item in report.failure_deltas
    )


def _route_metrics(
    checkpoint: OrchestrationCheckpoint,
    results: tuple[TaskResult, ...],
) -> tuple[bool, int]:
    first_result_by_task: dict[TaskId, TaskResult] = {}
    for result in results:
        first_result_by_task.setdefault(result.task_id, result)
    model_failures = {
        ErrorCode.MODEL_UNAVAILABLE,
        ErrorCode.MODEL_CAPABILITY_MISMATCH,
        ErrorCode.INVALID_MODEL_OUTPUT,
    }
    initial_valid = all(
        item.status is ResultStatus.SUCCEEDED
        or item.failure is None
        or item.failure.code not in model_failures
        for item in first_result_by_task.values()
    )
    fallback_count = max(0, len(checkpoint.execution_specs) - len(checkpoint.task_specs))
    return initial_valid, fallback_count


def _route_hard_constraints_satisfied(
    checkpoint: OrchestrationCheckpoint | None,
    profiles: tuple[ModelProfile, ...],
) -> bool | None:
    # 单一模型策略: 模型恒为 deepseek-v4-flash 且由固定路由器选择, 硬约束必然满足.
    # 保留返回值结构供未来多候选路由恢复。
    if checkpoint is None or not checkpoint.execution_specs:
        return None
    return all(
        item.model_routing.selected_model_key == "deepseek:deepseek-v4-flash"
        for item in checkpoint.execution_specs
    )


def _gateway_for_task(task: BenchmarkTask, gateway: ModelGateway) -> ModelGateway:
    if "model_fallback" in task.scenario_tags:
        return _OneShotUnavailableGateway(gateway)
    return gateway


def _system_acceptance(
    task: BenchmarkTask,
    checkpoint: OrchestrationCheckpoint,
    *,
    fallback_count: int,
) -> tuple[bool, tuple[str, ...]]:
    violations: list[str] = []
    if "model_fallback" in task.scenario_tags:
        # 单一模型策略: 模型失败必须结构化失败, 不允许回退链 (fallback_count 恒为 0)
        if fallback_count != 0:
            violations.append("单一模型策略不应产生任何回退实例")
    return not violations, tuple(violations)


def _completed_failure_reason(
    *,
    passed: bool,
    target_passed: bool,
    regression_passed: bool,
    system_violations: tuple[str, ...],
) -> str | None:
    if passed:
        return None
    reasons: list[str] = []
    if not target_passed:
        reasons.append("任务级隐藏验收未通过")
    if not regression_passed:
        reasons.append("出现新增回归或不稳定失败")
    reasons.extend(system_violations)
    return "；".join(reasons) or "完成记录未满足通过条件"


def _context_metrics(
    checkpoint: OrchestrationCheckpoint,
    artifact_store: LocalArtifactStore,
) -> tuple[int, int]:
    control_payload = json.dumps(
        {
            "graph": checkpoint.graph.model_dump(mode="json"),
            "result_summaries": [
                item.model_dump(mode="json") for item in checkpoint.result_summaries
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    builder = ContextBuilder(artifact_store)
    worker_chars = sum(len(builder.build(item).text) for item in checkpoint.execution_specs)
    return len(control_payload), worker_chars


def _invalid_task_count(checkpoint: OrchestrationCheckpoint) -> int:
    invalid = {TaskStatus.CANCELLED, TaskStatus.FAILED, TaskStatus.BLOCKED}
    return sum(item.status in invalid for item in checkpoint.graph.nodes)


def _usage_from_events(
    layout: RuntimeLayout,
    manifest: RunManifest,
) -> tuple[int, int, int]:
    events = JsonlEventRecorder(layout.events).events_for(manifest.run_id)
    model_calls = tuple(
        event for event in events if event.event_type is EventType.MODEL_CALL_COMPLETED
    )
    return (
        sum(_nonnegative_int(event.payload.get("input_tokens")) for event in model_calls),
        sum(_nonnegative_int(event.payload.get("output_tokens")) for event in model_calls),
        sum(_nonnegative_int(event.payload.get("reasoning_tokens")) for event in model_calls),
    )


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _failure_record(
    *,
    task: BenchmarkTask,
    manifest: RunManifest,
    plan: AdaptiveTaskPlan,
    agent_strategy: AgentStrategy,
    model_strategy: ModelStrategy,
    evidence_level: EvidenceLevel,
    predicted_difficulty: TaskDifficulty,
    duration_ms: int,
    checkpoint: OrchestrationCheckpoint | None,
    artifact_store: LocalArtifactStore,
    code: ErrorCode,
    message: str,
    suite_sha256: str,
    system_commit: str,
    usage: tuple[int, int, int],
    evidence_directory: str | None,
    model_profiles: tuple[ModelProfile, ...],
    trial_index: int,
) -> BenchmarkRunRecord:
    execution_specs = checkpoint.execution_specs if checkpoint is not None else ()
    context = _context_metrics(checkpoint, artifact_store) if checkpoint is not None else (0, 0)
    return BenchmarkRunRecord(
        benchmark_id=task.benchmark_id,
        run_id=str(manifest.run_id),
        trial_index=trial_index,
        suite_sha256=suite_sha256,
        system_commit=system_commit,
        agent_strategy=agent_strategy,
        model_strategy=model_strategy,
        evidence_level=evidence_level,
        status=BenchmarkRunStatus.FAILED,
        target_tests_passed=False,
        full_regression_passed=False,
        localization_candidates=tuple(
            dict.fromkeys(path for spec in plan.task_specs for path in spec.write_scope)
        ),
        patch_generated=False,
        patch_applied=False,
        patch_authorized=False,
        input_tokens=usage[0],
        output_tokens=usage[1],
        reasoning_tokens=usage[2],
        duration_ms=duration_ms,
        agent_count=len(execution_specs),
        task_count=len(plan.task_specs),
        invalid_task_count=(
            _invalid_task_count(checkpoint) if checkpoint is not None else len(plan.task_specs)
        ),
        conflict_count=1 if code is ErrorCode.PATCH_CONFLICT else 0,
        orchestrator_context_chars=context[0],
        worker_context_chars=context[1],
        initial_route_valid=code
        not in {
            ErrorCode.MODEL_UNAVAILABLE,
            ErrorCode.MODEL_CAPABILITY_MISMATCH,
            ErrorCode.INVALID_MODEL_OUTPUT,
        },
        route_hard_constraints_satisfied=_route_hard_constraints_satisfied(
            checkpoint,
            model_profiles,
        ),
        fallback_count=max(0, len(execution_specs) - len(plan.task_specs)),
        predicted_difficulty=predicted_difficulty,
        selected_model_keys=tuple(
            dict.fromkeys(item.model_routing.selected_model_key for item in execution_specs)
        ),
        evidence_directory=evidence_directory,
        failure_reason=f"{code.value}: {message}",
    )


def _record_key(
    record: BenchmarkRunRecord,
) -> tuple[str, str, str, AgentStrategy, ModelStrategy, EvidenceLevel, int]:
    return (
        record.suite_sha256,
        record.system_commit,
        record.benchmark_id,
        record.agent_strategy,
        record.model_strategy,
        record.evidence_level,
        record.trial_index,
    )


def _resolve_system_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(project_root), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode != 0:
        raise ValueError("无法解析 benchmark 系统 Git 提交")
    return commit


def _elapsed_ms(started: float) -> int:
    return max(0, int((monotonic() - started) * 1_000))


def _atomic_text(path: Path, content: str) -> None:
    _atomic_bytes(path, content.encode("utf-8"))


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
