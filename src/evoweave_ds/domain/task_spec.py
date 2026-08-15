"""Versioned execution contract for one independently schedulable task."""

from pydantic import Field, field_validator, model_validator

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import InputModality, RiskLevel
from evoweave_ds.domain.identifiers import ArtifactId, SpecId, TaskId
from evoweave_ds.domain.model_routing import DifficultyAssessment, ModelRequirement
from evoweave_ds.domain.validation import (
    validate_repository_path,
    validate_scope_subset,
    validate_unique_strings,
)


class TaskSpec(DomainModel):
    spec_id: SpecId
    task_id: TaskId
    change_spec_id: SpecId
    goal: str = Field(min_length=1, max_length=10_000)
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    acceptance_criteria: tuple[str, ...] = Field(min_length=1)
    depends_on: tuple[TaskId, ...] = ()
    input_artifact_ids: tuple[ArtifactId, ...] = ()
    context_artifact_ids: tuple[ArtifactId, ...] = ()
    read_scope: tuple[str, ...] = ()
    write_scope: tuple[str, ...] = ()
    required_modalities: tuple[InputModality, ...] = (InputModality.TEXT,)
    difficulty: DifficultyAssessment
    model_requirement: ModelRequirement
    risk_level: RiskLevel = RiskLevel.LOW
    version: int = Field(default=1, ge=1)

    @field_validator("read_scope", "write_scope")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "路径范围")

    @field_validator("acceptance_criteria")
    @classmethod
    def validate_acceptance_criteria(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return validate_unique_strings(values, "acceptance_criteria")

    @model_validator(mode="after")
    def validate_contract_consistency(self) -> "TaskSpec":
        if self.task_id in self.depends_on:
            raise ValueError("任务不能依赖自身")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("depends_on 不能重复")
        if len(set(self.input_artifact_ids)) != len(self.input_artifact_ids):
            raise ValueError("input_artifact_ids 不能重复")
        if len(set(self.context_artifact_ids)) != len(self.context_artifact_ids):
            raise ValueError("context_artifact_ids 不能重复")
        if self.difficulty.difficulty is not self.model_requirement.difficulty:
            raise ValueError("难度评估必须与模型需求一致")
        if self.model_requirement.task_id != self.task_id:
            raise ValueError("模型需求必须绑定当前任务")
        if set(self.required_modalities) != set(self.model_requirement.required_modalities):
            raise ValueError("任务与模型需求的输入模态必须一致")
        validate_scope_subset(
            self.write_scope,
            self.read_scope,
            child_name="write_scope",
            parent_name="read_scope",
        )
        return self
