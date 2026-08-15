"""Tests for identifiers, immutable models, events, and runtime limits."""

import importlib
import inspect
import pkgutil
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import evoweave_ds.domain as domain_package
from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import EventType, TaskDifficulty
from evoweave_ds.domain.events import DomainEvent
from evoweave_ds.domain.identifiers import EventId, RunId, TaskId
from evoweave_ds.domain.model_routing import ModelProfile, ModelRequirement, ModelRoutingDecision
from evoweave_ds.domain.resources import ResourceUsage, RuntimeLimits


def test_identifiers_have_type_specific_prefixes() -> None:
    assert RunId.new().startswith("run_")
    assert TaskId.new().startswith("task_")
    assert EventId.new().startswith("event_")


@pytest.mark.parametrize("value", ["wrong_abcdef", "task_short", "task_UPPERCASE"])
def test_identifier_rejects_invalid_value(value: str) -> None:
    with pytest.raises(ValueError, match="标识"):
        TaskId(value)


def test_domain_models_are_frozen() -> None:
    limits = RuntimeLimits()
    with pytest.raises(ValidationError):
        limits.max_steps = 10  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RuntimeLimits.model_validate({"max_steps": 10, "money_budget": 3})


def test_core_contracts_generate_json_schema() -> None:
    contract_types: set[type[DomainModel]] = set()
    for module_info in pkgutil.iter_modules(domain_package.__path__):
        module = importlib.import_module(f"{domain_package.__name__}.{module_info.name}")
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if (
                candidate is not DomainModel
                and candidate.__module__ == module.__name__
                and issubclass(candidate, DomainModel)
            ):
                contract_types.add(candidate)

    assert len(contract_types) >= 20
    for contract_type in contract_types:
        schema = contract_type.model_json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema


def test_model_requirement_has_no_price_or_monetary_budget_fields() -> None:
    forbidden = {"price", "cost", "budget", "money_budget", "max_cost"}
    for contract_type in (ModelRequirement, ModelProfile, ModelRoutingDecision):
        assert not set(contract_type.model_fields) & forbidden


def test_domain_event_round_trips_as_json() -> None:
    event = DomainEvent(
        event_id=EventId.new(),
        run_id=RunId.new(),
        sequence=1,
        event_type=EventType.RUN_CREATED,
        payload={"difficulty": TaskDifficulty.LOW.value},
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    restored = DomainEvent.model_validate_json(event.model_dump_json())
    assert restored == event


def test_resource_usage_respects_exact_limits() -> None:
    limits = RuntimeLimits(
        max_steps=2,
        max_input_tokens=100,
        max_output_tokens=50,
        max_tool_calls=1,
        timeout_seconds=3,
    )
    usage = ResourceUsage(
        steps=2,
        input_tokens=100,
        output_tokens=50,
        tool_calls=1,
        elapsed_ms=3_000,
    )
    assert not usage.exceeds(limits)
    assert usage.model_copy(update={"steps": 3}).exceeds(limits)


def test_cached_tokens_cannot_exceed_input_tokens() -> None:
    with pytest.raises(ValidationError, match="缓存命中"):
        ResourceUsage(input_tokens=10, cached_input_tokens=11)
