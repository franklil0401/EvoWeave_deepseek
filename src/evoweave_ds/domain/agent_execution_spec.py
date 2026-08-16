"""Immutable runtime configuration for one temporary worker instance."""

from pydantic import Field, field_validator, model_validator

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import InputModality
from evoweave_ds.domain.identifiers import AgentId, ArtifactId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelRoutingDecision
from evoweave_ds.domain.resources import RuntimeLimits
from evoweave_ds.domain.validation import (
    validate_repository_path,
    validate_scope_subset,
    validate_unique_strings,
)


class AgentExecutionSpec(DomainModel):
    spec_id: SpecId
    run_id: RunId
    agent_id: AgentId
    task_id: TaskId
    task_spec_id: SpecId
    task_spec_version: int = Field(ge=1)
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    goal: str = Field(min_length=1, max_length=10_000)
    acceptance_criteria: tuple[str, ...] = Field(min_length=1)
    required_modalities: tuple[InputModality, ...] = (InputModality.TEXT,)
    model_routing: ModelRoutingDecision
    tool_names: tuple[str, ...] = ()
    allowed_commands: tuple[str, ...] = ()
    read_scope: tuple[str, ...] = ()
    write_scope: tuple[str, ...] = ()
    context_artifact_ids: tuple[ArtifactId, ...] = ()
    input_artifact_ids: tuple[ArtifactId, ...] = ()
    runtime_limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    output_schema: str = Field(default="TaskResult", min_length=1, max_length=255)
    # 借鉴 dsh 可续接子代理(continuable): 默认 False 保持一次性 Worker 行为;
    # True 时允许 Orchestrator 通过 followup/resume 对同一 Worker 追加指令或带上下文重试。
    continuable: bool = False
    # 续接来源: parent_spec_id 指向本 Worker 的前一执行规格(retry/followup 链)。
    parent_spec_id: SpecId | None = None
    version: int = Field(default=1, ge=1)

    @field_validator("read_scope", "write_scope")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "路径范围")

    @field_validator("tool_names", "allowed_commands", "acceptance_criteria")
    @classmethod
    def validate_unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values):
            raise ValueError("执行规格列表不能包含空字符串")
        return validate_unique_strings(values, "执行规格列表")

    @model_validator(mode="after")
    def validate_execution_scope(self) -> "AgentExecutionSpec":
        validate_scope_subset(
            self.write_scope,
            self.read_scope,
            child_name="write_scope",
            parent_name="read_scope",
        )
        if len(set(self.context_artifact_ids)) != len(self.context_artifact_ids):
            raise ValueError("context_artifact_ids 不能重复")
        if len(set(self.input_artifact_ids)) != len(self.input_artifact_ids):
            raise ValueError("input_artifact_ids 不能重复")
        overlap = set(self.context_artifact_ids) & set(self.input_artifact_ids)
        if overlap:
            raise ValueError("同一产物不能同时作为 context 和 input")
        if not self.required_modalities:
            raise ValueError("required_modalities 不能为空")
        if len(set(self.required_modalities)) != len(self.required_modalities):
            raise ValueError("required_modalities 不能重复")
        if InputModality.TEXT not in self.required_modalities:
            raise ValueError("第一版执行规格必须包含 text 输入模态")
        return self
