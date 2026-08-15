from datetime import UTC, datetime
from pathlib import Path

import pytest

from evoweave_ds.domain.artifacts import EvidenceRef
from evoweave_ds.domain.enums import (
    EvidenceKind,
    InputModality,
    ModelAvailability,
    ModelTier,
    ResultStatus,
    TaskDifficulty,
    TaskStatus,
)
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import EvidenceId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import DifficultyAssessment, ModelProfile, ModelRequirement
from evoweave_ds.domain.policies import GraphPolicy
from evoweave_ds.domain.task_result import TaskFailure, TaskResult
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.infrastructure.models.fixed_router import FixedReasoningRouter
from evoweave_ds.infrastructure.persistence.graph_repository import SQLiteOrchestrationStore
from evoweave_ds.infrastructure.persistence.sqlite import SQLiteDatabase
from evoweave_ds.orchestration.agent_factory import AgentFactory, CapabilityPlan
from evoweave_ds.orchestration.checkpointing import CheckpointManager
from evoweave_ds.orchestration.decisions import (
    CancelTaskAction,
    FinishAction,
    OrchestratorDecision,
    RetryTaskAction,
    SplitTaskAction,
)
from evoweave_ds.orchestration.orchestrator import Orchestrator
from evoweave_ds.orchestration.scheduler import Scheduler
from evoweave_ds.orchestration.task_graph import TaskGraph


def test_scripted_dynamic_graph_expands_cancels_replaces_recovers_and_stops(
    tmp_path: Path,
) -> None:
    run_id = RunId.new()
    change_spec_id = SpecId.new()
    exploration = _task(
        change_spec_id=change_spec_id,
        goal="探索仓库并提出独立修改建议",
    )
    graph = TaskGraph.create(run_id=run_id, task_specs=(exploration,))
    policy = GraphPolicy(max_concurrent_tasks=2, max_no_progress_decisions=4)
    store = SQLiteOrchestrationStore(SQLiteDatabase(tmp_path / "runtime.db"))
    checkpoints = CheckpointManager(store)
    orchestrator = Orchestrator(
        graph=graph,
        graph_store=store,
        decision_ledger=store,
        checkpoint_manager=checkpoints,
        policy=policy,
    )
    factory = AgentFactory(
        model_router=FixedReasoningRouter(),
        model_profiles=(_profile(),),
    )
    scheduler = Scheduler(policy)

    exploration_execution = orchestrator.dispatch(
        scheduler=scheduler,
        agent_factory=factory,
        capability_plan_for=lambda _task_id: CapabilityPlan(tool_names=("file.read",)),
    )[0]
    orchestrator.accept_result(_result(exploration_execution, ResultStatus.SUCCEEDED))

    first_branch = _task(
        change_spec_id=change_spec_id,
        goal="修改独立模块 A",
        depends_on=(exploration.task_id,),
        write_scope=("src/a.py",),
    )
    second_branch = _task(
        change_spec_id=change_spec_id,
        goal="修改独立模块 B",
        depends_on=(exploration.task_id,),
        write_scope=("src/b.py",),
    )
    split = OrchestratorDecision(
        decision_id=SpecId.new(),
        run_id=run_id,
        based_on_graph_version=orchestrator.graph.snapshot.version,
        action=SplitTaskAction(
            source_task_id=exploration.task_id,
            task_specs=(first_branch, second_branch),
        ),
        rationale="探索结果给出两个独立写集合",
    )
    assert orchestrator.apply(split)
    parallel = orchestrator.dispatch(
        scheduler=scheduler,
        agent_factory=factory,
        capability_plan_for=lambda _task_id: CapabilityPlan(tool_names=("file.read", "file.write")),
    )
    assert len(parallel) == 2
    assert {item.task_id for item in parallel} == {
        first_branch.task_id,
        second_branch.task_id,
    }
    assert len(orchestrator.allocation_decisions[-1].selected_task_ids) == 2
    assert not hasattr(orchestrator.allocation_decisions[-1], "difficulty")

    checkpoint = checkpoints.load(run_id)
    assert checkpoint is not None
    restored = Orchestrator.restore(
        checkpoint,
        graph_store=store,
        decision_ledger=store,
        checkpoint_manager=checkpoints,
        policy=policy,
    )
    assert (
        restored.dispatch(
            scheduler=scheduler,
            agent_factory=factory,
            capability_plan_for=lambda _task_id: CapabilityPlan(),
        )
        == ()
    )

    first_execution = next(item for item in parallel if item.task_id == first_branch.task_id)
    second_execution = next(item for item in parallel if item.task_id == second_branch.task_id)
    cancel = OrchestratorDecision(
        decision_id=SpecId.new(),
        run_id=run_id,
        based_on_graph_version=restored.graph.snapshot.version,
        action=CancelTaskAction(
            task_id=first_branch.task_id,
            reason="新证据证明分支 A 无需修改",
        ),
        rationale="收缩无效分支",
    )
    restored.apply(cancel)
    assert restored.graph.node_for(first_execution.task_id).status is TaskStatus.CANCELLED

    restored.accept_result(_result(second_execution, ResultStatus.BLOCKED))
    previous_spec = restored.graph.spec_for(second_branch.task_id)
    replacement = previous_spec.model_copy(
        update={
            "spec_id": SpecId.new(),
            "goal": "使用新增命令能力修改独立模块 B",
            "model_requirement": previous_spec.model_requirement.model_copy(
                update={"version": previous_spec.model_requirement.version + 1}
            ),
            "version": previous_spec.version + 1,
        }
    )
    retry = OrchestratorDecision(
        decision_id=SpecId.new(),
        run_id=run_id,
        based_on_graph_version=restored.graph.snapshot.version,
        action=RetryTaskAction(
            task_id=second_branch.task_id,
            reason="需要增加命令能力",
            replacement_spec=replacement,
        ),
        rationale="生成新版本规格替换旧实例",
    )
    restored.apply(retry)
    replacement_execution = restored.dispatch(
        scheduler=scheduler,
        agent_factory=factory,
        capability_plan_for=lambda _task_id: CapabilityPlan(
            tool_names=("file.read", "file.write", "command.run"),
            allowed_commands=("python",),
        ),
    )[0]
    assert replacement_execution.task_spec_version == 2
    assert replacement_execution.version == 2
    assert replacement_execution.agent_id != second_execution.agent_id
    assert "command.run" in replacement_execution.tool_names

    restored.accept_result(_result(replacement_execution, ResultStatus.SUCCEEDED))
    restored.mark_acceptance_satisfied()
    finish = OrchestratorDecision(
        decision_id=SpecId.new(),
        run_id=run_id,
        based_on_graph_version=restored.graph.snapshot.version,
        action=FinishAction(summary="全部验收条件满足"),
        rationale="所有有效任务已完成",
    )
    assert restored.apply(finish)
    assert restored.finished
    assert restored.finish_summary == "全部验收条件满足"
    assert not restored.apply(finish)

    with pytest.raises(DomainError) as error:
        restored.apply(
            finish.model_copy(
                update={
                    "decision_id": SpecId.new(),
                    "based_on_graph_version": restored.graph.snapshot.version,
                }
            )
        )
    assert error.value.code is ErrorCode.INVALID_STATE_TRANSITION


def _task(
    *,
    change_spec_id: SpecId,
    goal: str,
    depends_on: tuple[TaskId, ...] = (),
    write_scope: tuple[str, ...] = (),
) -> TaskSpec:
    task_id = TaskId.new()
    return TaskSpec(
        spec_id=SpecId.new(),
        task_id=task_id,
        change_spec_id=change_spec_id,
        goal=goal,
        base_commit="a" * 40,
        acceptance_criteria=("完成",),
        depends_on=depends_on,
        read_scope=("src",),
        write_scope=write_scope,
        difficulty=DifficultyAssessment(difficulty=TaskDifficulty.LOW, rationale="测试"),
        model_requirement=ModelRequirement(
            requirement_id=SpecId.new(),
            task_id=task_id,
            difficulty=TaskDifficulty.LOW,
            required_modalities=(InputModality.TEXT,),
            min_context_tokens=1_000,
            min_output_tokens=500,
            requires_structured_output=True,
        ),
    )


def _profile() -> ModelProfile:
    return ModelProfile(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        tier=ModelTier.LOW,
        availability=ModelAvailability.AVAILABLE,
        input_modalities=(InputModality.TEXT,),
        context_window_tokens=8_000,
        max_output_tokens=2_000,
        supports_structured_output=True,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _result(execution, status: ResultStatus) -> TaskResult:
    failure = (
        TaskFailure(
            code=ErrorCode.CAPABILITY_NOT_FOUND,
            message="需要新增能力",
            retryable=True,
        )
        if status is not ResultStatus.SUCCEEDED
        else None
    )
    return TaskResult(
        task_id=execution.task_id,
        agent_id=execution.agent_id,
        execution_spec_id=execution.spec_id,
        execution_spec_version=execution.version,
        status=status,
        summary="任务结果",
        evidence=(
            EvidenceRef(
                evidence_id=EvidenceId.new(),
                kind=EvidenceKind.FILE,
                summary="结构化证据",
                repository_path="src/result.py",
            ),
        ),
        failure=failure,
    )
