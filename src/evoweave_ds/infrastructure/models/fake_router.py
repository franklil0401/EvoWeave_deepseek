"""Static offline router used to exercise routing contracts in stage 0."""

from collections.abc import Iterable

from evoweave_ds.domain.enums import ModelAvailability
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import SpecId
from evoweave_ds.domain.model_routing import (
    ModelCandidateRejection,
    ModelProfile,
    ModelRequirement,
    ModelRoutingDecision,
)


class StaticModelRouter:
    """Select from an explicit stable order after enforcing hard capabilities.

    The production difficulty-aware rule router belongs to stage 1. This test
    double exists so stage 0 can validate the port and decision schema without
    embedding provider behavior or prices in domain models.
    """

    def __init__(self, preferred_model_keys: Iterable[str]) -> None:
        self._preferred_model_keys = tuple(preferred_model_keys)

    def route(
        self,
        requirement: ModelRequirement,
        profiles: tuple[ModelProfile, ...],
    ) -> ModelRoutingDecision:
        profile_by_key = {profile.key: profile for profile in profiles}
        rejected: list[ModelCandidateRejection] = []
        eligible: dict[str, ModelProfile] = {}
        for profile in profiles:
            reasons = _rejection_reasons(requirement, profile)
            if reasons:
                rejected.append(
                    ModelCandidateRejection(model_key=profile.key, reasons=tuple(reasons))
                )
            else:
                eligible[profile.key] = profile

        ordered_keys = [
            key for key in self._preferred_model_keys if key in profile_by_key and key in eligible
        ]
        if not ordered_keys:
            raise DomainError(
                ErrorCode.MODEL_CAPABILITY_MISMATCH,
                "没有满足模型硬约束的静态候选",
                details={"task_id": str(requirement.task_id)},
            )
        selected_key, *fallbacks = ordered_keys
        selected = eligible[selected_key]
        return ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=requirement.requirement_id,
            requirement_version=requirement.version,
            selected_model_key=selected.key,
            selected_snapshot=selected.snapshot,
            reason="静态测试顺序中的第一个合格模型",
            fallback_model_keys=tuple(fallbacks),
            rejected_candidates=tuple(rejected),
            version=1,
        )


def _rejection_reasons(
    requirement: ModelRequirement,
    profile: ModelProfile,
) -> list[str]:
    reasons: list[str] = []
    if profile.availability is not ModelAvailability.AVAILABLE:
        reasons.append("模型当前不可用")
    if not set(requirement.required_modalities).issubset(profile.input_modalities):
        missing = sorted(
            modality.value
            for modality in set(requirement.required_modalities) - set(profile.input_modalities)
        )
        reasons.append(f"缺少输入模态：{','.join(missing)}")
    if profile.context_window_tokens < requirement.min_context_tokens:
        reasons.append("上下文窗口不足")
    if profile.max_output_tokens < requirement.min_output_tokens:
        reasons.append("最大输出不足")
    if requirement.requires_tool_calling and not profile.supports_tool_calling:
        reasons.append("不支持工具调用")
    if requirement.requires_structured_output and not profile.supports_structured_output:
        reasons.append("不支持结构化输出")
    if requirement.requires_thinking and not profile.supports_thinking:
        reasons.append("不支持思考模式")
    return reasons
