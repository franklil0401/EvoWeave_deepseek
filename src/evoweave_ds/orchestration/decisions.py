"""Finite, structured decisions available to the sole orchestrator."""

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.identifiers import RunId, SpecId, TaskId
from evoweave_ds.domain.task_spec import TaskSpec


class CreateTasksAction(DomainModel):
    action: Literal["create"] = "create"
    task_specs: tuple[TaskSpec, ...] = Field(min_length=1)


class SplitTaskAction(DomainModel):
    action: Literal["split"] = "split"
    source_task_id: TaskId
    task_specs: tuple[TaskSpec, ...] = Field(min_length=2)
    cancel_source: bool = False


class CancelTaskAction(DomainModel):
    action: Literal["cancel"] = "cancel"
    task_id: TaskId
    reason: str = Field(min_length=1, max_length=2_000)


class RetryTaskAction(DomainModel):
    action: Literal["retry"] = "retry"
    task_id: TaskId
    reason: str = Field(min_length=1, max_length=2_000)
    replacement_spec: TaskSpec | None = None


class ValidateTaskAction(DomainModel):
    action: Literal["validate"] = "validate"
    validated_task_id: TaskId
    validation_spec: TaskSpec


class WaitAction(DomainModel):
    action: Literal["wait"] = "wait"
    reason: str = Field(min_length=1, max_length=2_000)


class FinishAction(DomainModel):
    action: Literal["finish"] = "finish"
    summary: str = Field(min_length=1, max_length=10_000)


OrchestrationAction = Annotated[
    CreateTasksAction
    | SplitTaskAction
    | CancelTaskAction
    | RetryTaskAction
    | ValidateTaskAction
    | WaitAction
    | FinishAction,
    Field(discriminator="action"),
]

_ACTION_ADAPTER: TypeAdapter[OrchestrationAction] = TypeAdapter(OrchestrationAction)


class OrchestratorDecision(DomainModel):
    decision_id: SpecId
    run_id: RunId
    based_on_graph_version: int = Field(ge=1)
    action: OrchestrationAction
    rationale: str = Field(min_length=1, max_length=2_000)
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_embedded_run(self) -> "OrchestratorDecision":
        specs: tuple[TaskSpec, ...] = ()
        if isinstance(self.action, (CreateTasksAction, SplitTaskAction)):
            specs = self.action.task_specs
        elif isinstance(self.action, ValidateTaskAction):
            specs = (self.action.validation_spec,)
        elif isinstance(self.action, RetryTaskAction) and self.action.replacement_spec is not None:
            specs = (self.action.replacement_spec,)
        if len({spec.task_id for spec in specs}) != len(specs):
            raise ValueError("单个调度决策不能包含重复 task_id")
        return self


def orchestration_action_json_schema() -> dict[str, object]:
    return _ACTION_ADAPTER.json_schema()
