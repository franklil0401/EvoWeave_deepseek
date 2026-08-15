from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from evoweave_ds.application.analysis_service import AnalysisService
from evoweave_ds.application.configuration import EvoWeaveConfig
from evoweave_ds.application.intake_service import IntakeService
from evoweave_ds.application.run_state import JsonRunStateStore
from evoweave_ds.application.runtime_layout import RuntimeLayout
from evoweave_ds.benchmarking.models import AgentStrategy, ModelStrategy
from evoweave_ds.benchmarking.strategies import (
    planner_for_strategy,
    profiles_for_strategy,
    router_for_strategy,
)
from evoweave_ds.domain.enums import (
    InputModality,
    ModelAvailability,
    ModelTier,
    TaskDifficulty,
)
from evoweave_ds.domain.identifiers import SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelProfile, ModelRequirement
from evoweave_ds.infrastructure.artifacts.local_store import LocalArtifactStore


def test_agent_baselines_and_adaptive_planner_use_same_task_contract(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("multi_module")
    config = EvoWeaveConfig(runtime_directory=".runtime-strategies")
    layout = RuntimeLayout.create(repository, config)
    artifact_store = LocalArtifactStore(layout.artifacts)
    change = IntakeService().create(
        repository=repository,
        objective="同时更新客户模型与价格规则",
        acceptance_criteria=("回归通过",),
        allowed_paths=("src/shop/models.py", "src/shop/pricing.py"),
    )
    manifest, profile = AnalysisService(
        run_store=JsonRunStateStore(layout.run_state),
        artifact_store=artifact_store,
    ).analyze(change)

    plans = {
        strategy: planner_for_strategy(strategy, config).plan(manifest, profile)
        for strategy in AgentStrategy
    }

    assert len(plans[AgentStrategy.SINGLE].task_specs) == 1
    assert len(plans[AgentStrategy.FIXED_MULTI].task_specs) == 4
    assert len(plans[AgentStrategy.ADAPTIVE].task_specs) == 2
    fixed = plans[AgentStrategy.FIXED_MULTI].task_specs
    assert sum(bool(item.write_scope) for item in fixed) == 1
    assert [len(item.depends_on) for item in fixed] == [0, 1, 1, 1]
    assert all(type(item) is type(fixed[0]) for item in fixed)


def test_model_strategy_keeps_same_single_model_for_all_levels() -> None:
    profiles = (_flash_profile(),)

    low = profiles_for_strategy(ModelStrategy.FIXED_LOW, profiles)
    high = profiles_for_strategy(ModelStrategy.FIXED_HIGH, profiles)
    adaptive = profiles_for_strategy(ModelStrategy.ADAPTIVE, profiles)

    assert [item.model_id for item in low] == ["deepseek-v4-flash"]
    assert [item.model_id for item in high] == ["deepseek-v4-flash"]
    assert adaptive == profiles


def test_fixed_effort_router_fixes_reasoning_level_while_keeping_model() -> None:
    flash = _flash_profile()
    requirement = ModelRequirement(
        requirement_id=SpecId.new(),
        task_id=TaskId.new(),
        difficulty=TaskDifficulty.HIGH,
        required_modalities=(InputModality.TEXT,),
        min_context_tokens=64_000,
        min_output_tokens=8_000,
        requires_tool_calling=True,
        requires_structured_output=True,
    )

    low_decision = router_for_strategy(ModelStrategy.FIXED_LOW).route(requirement, (flash,))
    high_decision = router_for_strategy(ModelStrategy.FIXED_HIGH).route(requirement, (flash,))

    assert low_decision.selected_model_key == "deepseek:deepseek-v4-flash"
    assert low_decision.reasoning_effort == "low"
    assert "fixed_low" in low_decision.reason
    assert high_decision.reasoning_effort == "high"
    assert "fixed_high" in high_decision.reason


def _flash_profile() -> ModelProfile:
    return ModelProfile(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        tier=ModelTier.LOW,
        availability=ModelAvailability.AVAILABLE,
        input_modalities=(InputModality.TEXT,),
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_thinking=True,
        stable_priority=0,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
