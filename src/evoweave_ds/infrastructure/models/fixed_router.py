"""Fixed single-model router with difficulty-to-reasoning-effort mapping."""

from typing import Literal

from evoweave_ds.domain.base import utc_now
from evoweave_ds.domain.enums import ModelAvailability, TaskDifficulty
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import SpecId
from evoweave_ds.domain.model_routing import (
    ModelProfile,
    ModelRequirement,
    ModelRoutingDecision,
)

_REASONING_BY_DIFFICULTY: dict[TaskDifficulty, Literal["low", "medium", "high"]] = {
    TaskDifficulty.LOW: "low",
    TaskDifficulty.MEDIUM: "medium",
    TaskDifficulty.HIGH: "high",
}


class FixedReasoningRouter:
    """本阶段固定选择 deepseek-v4-flash, 并按任务难度映射推理等级 (low/medium/high)."""

    def route(
        self,
        requirement: ModelRequirement,
        profiles: tuple[ModelProfile, ...],
    ) -> ModelRoutingDecision:
        selected = next(
            (profile for profile in profiles if profile.key == "deepseek:deepseek-v4-flash"),
            None,
        )
        if selected is None or selected.availability is not ModelAvailability.AVAILABLE:
            raise DomainError(
                ErrorCode.MODEL_UNAVAILABLE,
                "未配置可用模型 deepseek:deepseek-v4-flash",
            )
        reasoning_effort = _REASONING_BY_DIFFICULTY[requirement.difficulty]
        return ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=requirement.requirement_id,
            requirement_version=requirement.version,
            selected_model_key=selected.key,
            selected_snapshot=selected.snapshot,
            reasoning_effort=reasoning_effort,
            reason=(
                f"固定模型 {selected.key}；任务难度 {requirement.difficulty.value} "
                f"映射推理等级 {reasoning_effort}"
            ),
            fallback_model_keys=(),
            rejected_candidates=(),
            capability_snapshot_at=selected.checked_at or utc_now(),
            version=1,
        )
