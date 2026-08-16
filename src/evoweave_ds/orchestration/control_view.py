"""Level-3 control projection that excludes source, logs, and artifact bytes."""

from pydantic import Field, field_validator, model_validator

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import ArtifactKind, ResultStatus, TaskStatus
from evoweave_ds.domain.identifiers import AgentId, ArtifactId, RunId, SpecId, TaskId
from evoweave_ds.domain.validation import validate_repository_path, validate_unique_strings


class ArtifactMetadata(DomainModel):
    artifact_id: ArtifactId
    kind: ArtifactKind
    media_type: str = Field(min_length=3, max_length=255)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TaskSuggestion(DomainModel):
    suggestion_id: SpecId
    source_task_id: TaskId
    goal: str = Field(min_length=1, max_length=2_000)
    rationale: str = Field(min_length=1, max_length=2_000)
    read_scope: tuple[str, ...] = ()
    write_scope: tuple[str, ...] = ()

    @field_validator("read_scope", "write_scope")
    @classmethod
    def validate_scopes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "任务建议范围")


class ResultControlSummary(DomainModel):
    task_id: TaskId
    agent_id: AgentId
    execution_spec_id: SpecId
    status: ResultStatus
    # 借鉴 dsh boundContextSummary: 控制视图只携带有界摘要(500 字符),
    # 完整轨迹留在事件/证据层, 需要时走 evidence.read 按需拉取。
    summary: str = Field(min_length=1, max_length=500)
    # 失败诊断(结构化): 失败命令、相关测试名、变更文件等, 让总调度不做
    # 证据拉取也能做二次决策。
    diagnostics: tuple[str, ...] = ()
    artifacts: tuple[ArtifactMetadata, ...] = ()
    suggestions: tuple[TaskSuggestion, ...] = ()


class TaskControlItem(DomainModel):
    task_id: TaskId
    task_spec_version: int = Field(ge=1)
    status: TaskStatus
    attempts: int = Field(ge=0)
    goal: str = Field(min_length=1, max_length=2_000)
    write_scope: tuple[str, ...] = ()


class OrchestrationControlView(DomainModel):
    run_id: RunId
    graph_version: int = Field(ge=1)
    tasks: tuple[TaskControlItem, ...]
    recent_results: tuple[ResultControlSummary, ...] = ()
    acceptance_satisfied: bool = False
    decision_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def reject_duplicate_tasks(self) -> "OrchestrationControlView":
        task_ids = [item.task_id for item in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("控制视图不能包含重复任务")
        return self
