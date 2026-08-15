"""Tests for change specs, task specs, and worker execution specs."""

import pytest
from pydantic import ValidationError

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.change_spec import ChangeSpec
from evoweave_ds.domain.enums import InputModality, TaskDifficulty
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import (
    DifficultyAssessment,
    ModelRequirement,
    ModelRoutingDecision,
)
from evoweave_ds.domain.task_spec import TaskSpec

BASE_COMMIT = "a" * 40


def _requirement(
    task_id: TaskId,
    modalities: tuple[InputModality, ...] = (InputModality.TEXT,),
) -> ModelRequirement:
    return ModelRequirement(
        requirement_id=SpecId.new(),
        task_id=task_id,
        difficulty=TaskDifficulty.LOW,
        required_modalities=modalities,
    )


def _task_spec(
    *,
    task_id: TaskId,
    modalities: tuple[InputModality, ...] = (InputModality.TEXT,),
    read_scope: tuple[str, ...] = ("src",),
    write_scope: tuple[str, ...] = ("src/evoweave_ds",),
) -> TaskSpec:
    return TaskSpec(
        spec_id=SpecId.new(),
        task_id=task_id,
        change_spec_id=SpecId.new(),
        goal="修改并验证领域协议",
        base_commit=BASE_COMMIT,
        acceptance_criteria=("测试通过",),
        read_scope=read_scope,
        write_scope=write_scope,
        required_modalities=modalities,
        difficulty=DifficultyAssessment(
            difficulty=TaskDifficulty.LOW,
            rationale="局部、低风险修改",
        ),
        model_requirement=_requirement(task_id, modalities),
    )


@pytest.mark.parametrize("missing_field", ["objective", "acceptance_criteria", "base_commit"])
def test_change_spec_rejects_missing_core_contract_field(missing_field: str) -> None:
    data: dict[str, object] = {
        "spec_id": SpecId.new(),
        "run_id": RunId.new(),
        "objective": "更新软件",
        "repository": "local/repository",
        "base_commit": BASE_COMMIT,
        "acceptance_criteria": ("测试通过",),
    }
    del data[missing_field]
    with pytest.raises(ValidationError):
        ChangeSpec.model_validate(data)


def test_write_scope_may_be_nested_below_read_scope() -> None:
    spec = _task_spec(task_id=TaskId.new())
    assert spec.write_scope == ("src/evoweave_ds",)


def test_write_scope_cannot_escape_read_scope() -> None:
    with pytest.raises(ValidationError, match="write_scope"):
        _task_spec(
            task_id=TaskId.new(),
            read_scope=("src",),
            write_scope=("tests",),
        )


@pytest.mark.parametrize("missing_field", ["goal", "acceptance_criteria", "base_commit"])
def test_task_spec_rejects_missing_core_contract_field(missing_field: str) -> None:
    task_id = TaskId.new()
    data: dict[str, object] = {
        "spec_id": SpecId.new(),
        "task_id": task_id,
        "change_spec_id": SpecId.new(),
        "goal": "更新领域模型",
        "base_commit": BASE_COMMIT,
        "acceptance_criteria": ("测试通过",),
        "read_scope": ("src",),
        "difficulty": DifficultyAssessment(
            difficulty=TaskDifficulty.LOW,
            rationale="局部修改",
        ),
        "model_requirement": _requirement(task_id),
    }
    del data[missing_field]
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(data)


def test_agent_execution_spec_is_version_pinned() -> None:
    task_id = TaskId.new()
    decision = ModelRoutingDecision(
        decision_id=SpecId.new(),
        requirement_id=SpecId.new(),
        requirement_version=2,
        selected_model_key="fake:text-small",
        reasoning_effort="low",
        reason="满足硬能力约束",
    )
    spec = AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=task_id,
        task_spec_id=SpecId.new(),
        task_spec_version=3,
        base_commit="a" * 40,
        goal="更新领域协议",
        acceptance_criteria=("测试通过",),
        model_routing=decision,
        read_scope=("src",),
        write_scope=("src/evoweave_ds",),
    )
    assert spec.task_spec_version == 3
    assert spec.model_routing.requirement_version == 2
