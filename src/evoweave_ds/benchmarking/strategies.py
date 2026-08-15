"""Benchmark-only Agent and reasoning-effort strategy controls."""

from typing import Literal

from evoweave_ds.application.adaptive_task_planner import (
    AdaptiveTaskPlan,
    AdaptiveTaskPlanner,
    TaskPlanner,
)
from evoweave_ds.application.configuration import EvoWeaveConfig
from evoweave_ds.benchmarking.models import AgentStrategy, ModelStrategy
from evoweave_ds.domain.base import utc_now
from evoweave_ds.domain.enums import InputModality
from evoweave_ds.domain.identifiers import SpecId, TaskId
from evoweave_ds.domain.model_routing import (
    ModelProfile,
    ModelRequirement,
    ModelRoutingDecision,
)
from evoweave_ds.domain.ports import ModelRouter
from evoweave_ds.domain.repository_models import RepositoryProfile
from evoweave_ds.domain.run_models import RunManifest
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.infrastructure.models.fixed_router import FixedReasoningRouter

_FIXED_MODEL_KEY = "deepseek:deepseek-v4-flash"


class FixedReasoningEffortRouter:
    """Fix reasoning_effort to one level while keeping the model fixed to flash."""

    def __init__(self, strategy: ModelStrategy) -> None:
        if strategy not in {ModelStrategy.FIXED_LOW, ModelStrategy.FIXED_HIGH}:
            raise ValueError("固定推理等级策略只支持 fixed_low / fixed_high")
        self._effort: Literal["low", "high"] = (
            "low" if strategy is ModelStrategy.FIXED_LOW else "high"
        )
        self._strategy = strategy

    def route(
        self,
        requirement: ModelRequirement,
        profiles: tuple[ModelProfile, ...],
    ) -> ModelRoutingDecision:
        selected = next(
            (profile for profile in profiles if profile.key == _FIXED_MODEL_KEY),
            None,
        )
        if selected is None:
            raise ValueError("缺少固定模型 deepseek:deepseek-v4-flash")
        return ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=requirement.requirement_id,
            requirement_version=requirement.version,
            selected_model_key=selected.key,
            selected_snapshot=selected.snapshot,
            reasoning_effort=self._effort,
            reason=(
                f"benchmark {self._strategy.value} 对照固定推理等级 {self._effort}；"
                "模型恒为 deepseek-v4-flash"
            ),
            fallback_model_keys=(),
            rejected_candidates=(),
            capability_snapshot_at=selected.checked_at or utc_now(),
            version=1,
        )


class SingleAgentTaskPlanner:
    def __init__(self, config: EvoWeaveConfig) -> None:
        self._delegate = AdaptiveTaskPlanner(config.model_copy(update={"max_dynamic_tasks": 1}))

    def plan(
        self,
        manifest: RunManifest,
        profile: RepositoryProfile,
    ) -> AdaptiveTaskPlan:
        base = self._delegate.plan(manifest, profile)
        return AdaptiveTaskPlan(
            task_specs=base.task_specs,
            rationale="单 Agent 对照：所有允许写范围合并为一个通用执行实例",
        )


class FixedMultiTaskPlanner:
    """Four-step fixed baseline; task functions are data, not product role classes."""

    def __init__(self, config: EvoWeaveConfig) -> None:
        self._single = SingleAgentTaskPlanner(config)

    def plan(
        self,
        manifest: RunManifest,
        profile: RepositoryProfile,
    ) -> AdaptiveTaskPlan:
        implementation_source = self._single.plan(manifest, profile).task_specs[0]
        exploration = _derived_task(
            implementation_source,
            goal="固定流水线步骤 1/4：读取仓库并总结需求相关证据，不修改文件",
            write_scope=(),
            depends_on=(),
            keep_input_modalities=True,
        )
        implementation = _derived_task(
            implementation_source,
            goal=(
                "固定流水线步骤 2/4：根据用户目标修改全部授权写范围；"
                f"原目标：{manifest.change_spec.objective}"
            ),
            write_scope=implementation_source.write_scope,
            depends_on=(exploration.task_id,),
            keep_input_modalities=True,
        )
        review = _derived_task(
            implementation_source,
            goal="固定流水线步骤 3/4：只读审查修改范围和验收条件，不修改文件",
            write_scope=(),
            depends_on=(implementation.task_id,),
            keep_input_modalities=False,
        )
        validation = _derived_task(
            implementation_source,
            goal="固定流水线步骤 4/4：只读检查测试与风险信息，不修改文件",
            write_scope=(),
            depends_on=(review.task_id,),
            keep_input_modalities=False,
        )
        return AdaptiveTaskPlan(
            task_specs=(exploration, implementation, review, validation),
            rationale="固定多 Agent 对照：始终创建四个串行通用实例",
        )


def planner_for_strategy(
    strategy: AgentStrategy,
    config: EvoWeaveConfig,
) -> TaskPlanner:
    if strategy is AgentStrategy.SINGLE:
        return SingleAgentTaskPlanner(config)
    if strategy is AgentStrategy.FIXED_MULTI:
        return FixedMultiTaskPlanner(config)
    return AdaptiveTaskPlanner(config)


def profiles_for_strategy(
    strategy: ModelStrategy,
    profiles: tuple[ModelProfile, ...],
) -> tuple[ModelProfile, ...]:
    # 单一模型策略: 任何推理等级策略都使用同一模型目录 (deepseek-v4-flash)
    return profiles


def router_for_strategy(strategy: ModelStrategy) -> ModelRouter:
    if strategy is ModelStrategy.ADAPTIVE:
        return FixedReasoningRouter()
    return FixedReasoningEffortRouter(strategy)


def _derived_task(
    source: TaskSpec,
    *,
    goal: str,
    write_scope: tuple[str, ...],
    depends_on: tuple[TaskId, ...],
    keep_input_modalities: bool,
) -> TaskSpec:
    task_id = TaskId.new()
    modalities = source.required_modalities if keep_input_modalities else (InputModality.TEXT,)
    input_artifact_ids = source.input_artifact_ids if keep_input_modalities else ()
    requirement = ModelRequirement.model_validate(
        {
            **source.model_requirement.model_dump(),
            "requirement_id": SpecId.new(),
            "task_id": task_id,
            "required_modalities": modalities,
        }
    )
    return TaskSpec(
        spec_id=SpecId.new(),
        task_id=task_id,
        change_spec_id=source.change_spec_id,
        goal=goal,
        base_commit=source.base_commit,
        acceptance_criteria=source.acceptance_criteria,
        depends_on=depends_on,
        input_artifact_ids=input_artifact_ids,
        context_artifact_ids=source.context_artifact_ids,
        read_scope=source.read_scope,
        write_scope=write_scope,
        required_modalities=modalities,
        difficulty=source.difficulty,
        model_requirement=requirement,
        risk_level=source.risk_level,
    )
