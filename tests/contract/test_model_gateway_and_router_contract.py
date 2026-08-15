"""Contract tests for offline model access and hard-capability routing."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from evoweave_ds.domain.enums import (
    InputModality,
    ModelAvailability,
    ModelTier,
    TaskDifficulty,
)
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelProfile, ModelRequirement, ModelRoutingDecision
from evoweave_ds.domain.ports import ModelGateway, ModelRequest, ModelResponse
from evoweave_ds.infrastructure.models.fake import ScriptedModelGateway
from evoweave_ds.infrastructure.models.fake_router import StaticModelRouter


def _profile(
    model_id: str,
    *,
    modalities: tuple[InputModality, ...],
    availability: ModelAvailability = ModelAvailability.AVAILABLE,
    context_tokens: int = 32_000,
) -> ModelProfile:
    return ModelProfile(
        provider="fake",
        model_id=model_id,
        tier=ModelTier.MEDIUM,
        availability=availability,
        input_modalities=modalities,
        context_window_tokens=context_tokens,
        max_output_tokens=8_000,
        supports_tool_calling=True,
        supports_structured_output=True,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC)
        if availability is ModelAvailability.AVAILABLE
        else None,
    )


def _requirement(modalities: tuple[InputModality, ...]) -> ModelRequirement:
    return ModelRequirement(
        requirement_id=SpecId.new(),
        task_id=TaskId.new(),
        difficulty=TaskDifficulty.MEDIUM,
        required_modalities=modalities,
        min_context_tokens=16_000,
        min_output_tokens=2_000,
        requires_structured_output=True,
    )


def test_scripted_gateway_implements_port_and_records_request() -> None:
    profile = _profile("text", modalities=(InputModality.TEXT,))
    response = ModelResponse(model_key=profile.key, text="ok", input_tokens=4, output_tokens=1)
    gateway = ScriptedModelGateway(profiles=(profile,), responses=(response,))
    assert isinstance(gateway, ModelGateway)
    request = ModelRequest(model_key=profile.key, messages=("hello",), max_output_tokens=10)
    assert gateway.complete(request) == response
    assert gateway.requests == [request]
    assert gateway.list_profiles() == (profile,)


def test_scripted_gateway_fails_when_script_is_exhausted() -> None:
    gateway = ScriptedModelGateway()
    with pytest.raises(DomainError) as error:
        gateway.complete(
            ModelRequest(model_key="fake:text", messages=("hello",), max_output_tokens=10)
        )
    assert error.value.code is ErrorCode.SCRIPT_EXHAUSTED


def test_router_records_eligible_fallbacks_in_stable_order() -> None:
    first = _profile("first", modalities=(InputModality.TEXT,))
    second = _profile("second", modalities=(InputModality.TEXT,))
    router = StaticModelRouter((second.key, first.key))
    decision = router.route(_requirement((InputModality.TEXT,)), (first, second))
    assert decision.selected_model_key == second.key
    assert decision.fallback_model_keys == (first.key,)


def test_router_never_selects_unavailable_model() -> None:
    unavailable = _profile(
        "offline",
        modalities=(InputModality.TEXT,),
        availability=ModelAvailability.UNAVAILABLE,
    )
    available = _profile("online", modalities=(InputModality.TEXT,))
    router = StaticModelRouter((unavailable.key, available.key))
    decision = router.route(_requirement((InputModality.TEXT,)), (unavailable, available))
    assert decision.selected_model_key == available.key
    assert decision.rejected_candidates[0].model_key == unavailable.key


def test_router_selection_is_deterministic_for_same_snapshot() -> None:
    first = _profile("first", modalities=(InputModality.TEXT,))
    second = _profile("second", modalities=(InputModality.TEXT,))
    requirement = _requirement((InputModality.TEXT,))
    router = StaticModelRouter((first.key, second.key))
    decisions = [router.route(requirement, (second, first)) for _ in range(2)]
    assert [decision.selected_model_key for decision in decisions] == [first.key, first.key]
    assert [decision.fallback_model_keys for decision in decisions] == [
        (second.key,),
        (second.key,),
    ]


def test_router_rejects_when_no_model_satisfies_hard_constraints() -> None:
    text = _profile("text", modalities=(InputModality.TEXT,), context_tokens=8_000)
    router = StaticModelRouter((text.key,))
    with pytest.raises(DomainError) as error:
        router.route(
            _requirement((InputModality.TEXT,)),
            (text,),
        )
    assert error.value.code is ErrorCode.MODEL_CAPABILITY_MISMATCH


def test_available_model_profile_requires_observation_time() -> None:
    with pytest.raises(ValueError, match="checked_at"):
        ModelProfile(
            provider="fake",
            model_id="unknown",
            tier=ModelTier.LOW,
            availability=ModelAvailability.AVAILABLE,
            input_modalities=(InputModality.TEXT,),
            context_window_tokens=1,
            max_output_tokens=1,
        )


def test_routing_decision_cannot_claim_unavailable_selection() -> None:
    with pytest.raises(ValidationError, match="selected_availability"):
        ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=SpecId.new(),
            requirement_version=1,
            selected_model_key="fake:offline",
            selected_availability=ModelAvailability.UNAVAILABLE,
            reason="无效示例",
        )
