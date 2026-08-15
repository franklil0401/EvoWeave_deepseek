"""Tests for explainable difficulty and fixed-model reasoning-effort routing."""

from datetime import UTC, datetime

import pytest

from evoweave_ds.agent_runtime.difficulty import RuleBasedDifficultyAssessor, TaskSignals
from evoweave_ds.domain.enums import (
    InputModality,
    ModelAvailability,
    ModelTier,
    RiskLevel,
    TaskDifficulty,
)
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelProfile, ModelRequirement
from evoweave_ds.infrastructure.models.fixed_router import FixedReasoningRouter

CHECKED_AT = datetime(2026, 8, 14, tzinfo=UTC)


def _profile(
    model_id: str,
    tier: ModelTier,
    *,
    availability: ModelAvailability = ModelAvailability.AVAILABLE,
    priority: int = 100,
    provider: str = "deepseek",
) -> ModelProfile:
    return ModelProfile(
        provider=provider,
        model_id=model_id,
        tier=tier,
        availability=availability,
        input_modalities=(InputModality.TEXT,),
        context_window_tokens=100_000,
        max_output_tokens=8_000,
        supports_tool_calling=True,
        supports_structured_output=True,
        checked_at=CHECKED_AT if availability is ModelAvailability.AVAILABLE else None,
        stable_priority=priority,
    )


def _flash_profile() -> ModelProfile:
    return _profile("deepseek-v4-flash", ModelTier.LOW, priority=0)


def _requirement(difficulty: TaskDifficulty) -> ModelRequirement:
    return ModelRequirement(
        requirement_id=SpecId.new(),
        task_id=TaskId.new(),
        difficulty=difficulty,
        required_modalities=(InputModality.TEXT,),
        min_context_tokens=1_000,
        min_output_tokens=500,
        requires_tool_calling=True,
        requires_structured_output=True,
    )


def test_difficulty_assessor_keeps_small_local_change_low() -> None:
    assessment = RuleBasedDifficultyAssessor().assess(
        TaskSignals(affected_files=1, affected_symbols=2)
    )
    assert assessment.difficulty is TaskDifficulty.LOW
    assert "规则评分" in assessment.rationale


def test_difficulty_assessor_marks_unknown_high_risk_change_high() -> None:
    assessment = RuleBasedDifficultyAssessor().assess(
        TaskSignals(
            affected_files=4,
            dependency_depth=4,
            crosses_modules=True,
            scope_is_unknown=True,
            risk_level=RiskLevel.HIGH,
        )
    )
    assert assessment.difficulty is TaskDifficulty.HIGH
    assert "影响范围未知" in assessment.rationale


@pytest.mark.parametrize(
    ("difficulty", "expected_effort"),
    [
        (TaskDifficulty.LOW, "low"),
        (TaskDifficulty.MEDIUM, "medium"),
        (TaskDifficulty.HIGH, "high"),
    ],
)
def test_fixed_router_maps_difficulty_to_reasoning_effort(
    difficulty: TaskDifficulty,
    expected_effort: str,
) -> None:
    decision = FixedReasoningRouter().route(
        _requirement(difficulty),
        (_flash_profile(),),
    )
    assert decision.selected_model_key == "deepseek:deepseek-v4-flash"
    assert decision.reasoning_effort == expected_effort
    assert decision.fallback_model_keys == ()
    assert "deepseek-v4-flash" in decision.reason


def test_fixed_router_never_changes_model_for_high_difficulty() -> None:
    decision = FixedReasoningRouter().route(
        _requirement(TaskDifficulty.HIGH),
        (_flash_profile(),),
    )
    assert decision.selected_model_key == "deepseek:deepseek-v4-flash"


def test_fixed_router_rejects_when_flash_missing() -> None:
    with pytest.raises(DomainError) as error:
        FixedReasoningRouter().route(
            _requirement(TaskDifficulty.LOW),
            (_profile("other-model", ModelTier.HIGH),),
        )
    assert error.value.code is ErrorCode.MODEL_UNAVAILABLE


def test_fixed_router_rejects_when_flash_unavailable() -> None:
    offline = _flash_profile().model_copy(
        update={"availability": ModelAvailability.UNAVAILABLE, "checked_at": None}
    )
    with pytest.raises(DomainError) as error:
        FixedReasoningRouter().route(_requirement(TaskDifficulty.LOW), (offline,))
    assert error.value.code is ErrorCode.MODEL_UNAVAILABLE


def test_fixed_router_semantics_are_deterministic() -> None:
    router = FixedReasoningRouter()
    requirement = _requirement(TaskDifficulty.MEDIUM)
    decisions = [router.route(requirement, (_flash_profile(),)) for _ in range(3)]
    assert [item.selected_model_key for item in decisions] == [
        "deepseek:deepseek-v4-flash",
    ] * 3
    assert [item.reasoning_effort for item in decisions] == ["medium"] * 3
    assert all(item.capability_snapshot_at == CHECKED_AT for item in decisions)
