"""Create capability-scoped temporary workers only after explicit model routing."""

from pydantic import Field, field_validator

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId
from evoweave_ds.domain.model_routing import ModelProfile
from evoweave_ds.domain.ports import ModelRouter
from evoweave_ds.domain.resources import RuntimeLimits
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.domain.validation import validate_unique_strings


class CapabilityPlan(DomainModel):
    tool_names: tuple[str, ...] = ()
    allowed_commands: tuple[str, ...] = ()
    runtime_limits: RuntimeLimits = Field(default_factory=RuntimeLimits)

    @field_validator("tool_names", "allowed_commands")
    @classmethod
    def validate_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return validate_unique_strings(values, "能力计划")


class AgentFactory:
    def __init__(
        self,
        *,
        model_router: ModelRouter,
        model_profiles: tuple[ModelProfile, ...],
    ) -> None:
        self._router = model_router
        self._profiles = model_profiles

    def create(
        self,
        *,
        run_id: RunId,
        task_spec: TaskSpec,
        capability_plan: CapabilityPlan,
        version: int = 1,
    ) -> AgentExecutionSpec:
        routing = self._router.route(task_spec.model_requirement, self._profiles)
        return AgentExecutionSpec(
            spec_id=SpecId.new(),
            run_id=run_id,
            agent_id=AgentId.new(),
            task_id=task_spec.task_id,
            task_spec_id=task_spec.spec_id,
            task_spec_version=task_spec.version,
            base_commit=task_spec.base_commit,
            goal=task_spec.goal,
            acceptance_criteria=task_spec.acceptance_criteria,
            required_modalities=task_spec.required_modalities,
            model_routing=routing,
            tool_names=capability_plan.tool_names,
            allowed_commands=capability_plan.allowed_commands,
            read_scope=task_spec.read_scope,
            write_scope=task_spec.write_scope,
            context_artifact_ids=task_spec.context_artifact_ids,
            input_artifact_ids=task_spec.input_artifact_ids,
            runtime_limits=capability_plan.runtime_limits,
            version=version,
        )
