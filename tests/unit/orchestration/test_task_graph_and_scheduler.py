from datetime import UTC, datetime

from evoweave_ds.domain.enums import (
    InputModality,
    ModelAvailability,
    ModelTier,
    TaskDifficulty,
    TaskStatus,
)
from evoweave_ds.domain.identifiers import ArtifactId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import DifficultyAssessment, ModelProfile, ModelRequirement
from evoweave_ds.domain.policies import GraphPolicy
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.infrastructure.models.fixed_router import FixedReasoningRouter
from evoweave_ds.orchestration.agent_factory import AgentFactory, CapabilityPlan
from evoweave_ds.orchestration.scheduler import Scheduler, write_scopes_overlap
from evoweave_ds.orchestration.task_graph import TaskGraph


def test_task_graph_readiness_and_version_history() -> None:
    change_spec_id = SpecId.new()
    first = _task(change_spec_id=change_spec_id, goal="探索仓库")
    second = _task(
        change_spec_id=change_spec_id,
        goal="修改服务",
        depends_on=(first.task_id,),
    )
    graph = TaskGraph.create(run_id=RunId.new(), task_specs=(first, second))

    assert graph.node_for(first.task_id).status is TaskStatus.READY
    assert graph.node_for(second.task_id).status is TaskStatus.CREATED
    graph.transition(first.task_id, TaskStatus.LEASED)
    graph.transition(first.task_id, TaskStatus.RUNNING)
    graph.transition(first.task_id, TaskStatus.SUCCEEDED)
    assert graph.node_for(second.task_id).status is TaskStatus.READY
    assert [snapshot.version for snapshot in graph.history] == list(
        range(1, graph.snapshot.version + 1)
    )
    assert len(graph.version_records) == len(graph.history)


def test_scheduler_parallelizes_independent_scopes_but_not_overlapping_writes() -> None:
    change_spec_id = SpecId.new()
    first = _task(
        change_spec_id=change_spec_id,
        goal="修改模块 A",
        write_scope=("src/a",),
    )
    second = _task(
        change_spec_id=change_spec_id,
        goal="修改模块 B",
        write_scope=("src/b",),
    )
    graph = TaskGraph.create(run_id=RunId.new(), task_specs=(first, second))
    scheduler = Scheduler(GraphPolicy(max_concurrent_tasks=4))

    assert set(scheduler.select_ready(graph)) == {first.task_id, second.task_id}

    overlapping = _task(
        change_spec_id=change_spec_id,
        goal="修改模块 A 子目录",
        write_scope=("src/a/internal",),
    )
    overlap_graph = TaskGraph.create(
        run_id=RunId.new(),
        task_specs=(first, overlapping),
    )
    assert len(scheduler.select_ready(overlap_graph)) == 1
    assert write_scopes_overlap(("src/a",), ("src/a/internal",)) is True


def test_high_difficulty_does_not_create_extra_agent_instances() -> None:
    high = _task(
        change_spec_id=SpecId.new(),
        goal="不可拆分的高风险任务",
        difficulty=TaskDifficulty.HIGH,
        write_scope=("src/core",),
    )
    graph = TaskGraph.create(run_id=RunId.new(), task_specs=(high,))

    assert Scheduler(GraphPolicy(max_concurrent_tasks=8)).select_ready(graph) == (high.task_id,)


def test_agent_factory_maps_difficulty_to_reasoning_effort() -> None:
    task = _task(
        change_spec_id=SpecId.new(),
        goal="完成高风险重构",
        difficulty=TaskDifficulty.HIGH,
        write_scope=("src/ui",),
    )
    factory = AgentFactory(
        model_router=FixedReasoningRouter(),
        model_profiles=_profiles(),
    )
    execution = factory.create(
        run_id=RunId.new(),
        task_spec=task,
        capability_plan=CapabilityPlan(tool_names=("file.read", "file.write")),
    )

    assert execution.required_modalities == (InputModality.TEXT,)
    assert execution.model_routing.selected_model_key == "deepseek:deepseek-v4-flash"
    assert execution.model_routing.reasoning_effort == "high"
    assert execution.tool_names == ("file.read", "file.write")


def _task(
    *,
    change_spec_id: SpecId,
    goal: str,
    depends_on: tuple[TaskId, ...] = (),
    write_scope: tuple[str, ...] = (),
    difficulty: TaskDifficulty = TaskDifficulty.LOW,
    modalities: tuple[InputModality, ...] = (InputModality.TEXT,),
    input_artifact_ids: tuple[ArtifactId, ...] = (),
    version: int = 1,
    task_id: TaskId | None = None,
) -> TaskSpec:
    resolved_task_id = task_id or TaskId.new()
    requirement = ModelRequirement(
        requirement_id=SpecId.new(),
        task_id=resolved_task_id,
        difficulty=difficulty,
        required_modalities=modalities,
        min_context_tokens=1_000,
        min_output_tokens=500,
        requires_structured_output=True,
        version=version,
    )
    return TaskSpec(
        spec_id=SpecId.new(),
        task_id=resolved_task_id,
        change_spec_id=change_spec_id,
        goal=goal,
        base_commit="a" * 40,
        acceptance_criteria=("完成",),
        depends_on=depends_on,
        input_artifact_ids=input_artifact_ids,
        read_scope=("src",),
        write_scope=write_scope,
        required_modalities=modalities,
        difficulty=DifficultyAssessment(difficulty=difficulty, rationale="测试"),
        model_requirement=requirement,
        version=version,
    )


def _profiles() -> tuple[ModelProfile, ...]:
    checked_at = datetime(2026, 1, 1, tzinfo=UTC)
    return (
        ModelProfile(
            provider="deepseek",
            model_id="deepseek-v4-flash",
            tier=ModelTier.LOW,
            availability=ModelAvailability.AVAILABLE,
            input_modalities=(InputModality.TEXT,),
            context_window_tokens=1_000_000,
            max_output_tokens=128_000,
            supports_structured_output=True,
            checked_at=checked_at,
        ),
    )
