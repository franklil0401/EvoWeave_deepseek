"""Create capability-scoped temporary workers only after explicit model routing."""

from typing import Protocol

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


class WorkerProvider(Protocol):
    """执行后端缝隙(借鉴 dsh subagent provider 注册表)。

    默认 in-process; 未来可注册 fork/external(dsh SDK 桥接) 等后端。
    """

    @property
    def name(self) -> str: ...

    def register(self, spec: AgentExecutionSpec) -> None: ...

    def unregister(self, spec_id: SpecId) -> None: ...


class InProcessWorkerProvider:
    """默认同进程执行后端: 只维护已发布 spec 的注册表, 执行仍走 WorkerRuntime。"""

    def __init__(self) -> None:
        self._registered: dict[SpecId, AgentExecutionSpec] = {}

    @property
    def name(self) -> str:
        return "in-process"

    def register(self, spec: AgentExecutionSpec) -> None:
        self._registered[spec.spec_id] = spec

    def unregister(self, spec_id: SpecId) -> None:
        self._registered.pop(spec_id, None)

    def get(self, spec_id: SpecId) -> AgentExecutionSpec | None:
        return self._registered.get(spec_id)


class AgentFactory:
    """Create capability-scoped execution specs through an explicit two-phase
    transaction (begin/commit), with a worker-provider registry seam.

    借鉴 dsh: 创建事务化 — begin 产出未发布 spec, commit 前 Orchestrator
    永远看不到半配置 Worker; rollback 清理。create() 保持旧语义(等价于
    begin+commit), 不破坏既有调用方。
    """

    def __init__(
        self,
        *,
        model_router: ModelRouter,
        model_profiles: tuple[ModelProfile, ...],
        worker_provider: WorkerProvider | None = None,
    ) -> None:
        self._router = model_router
        self._profiles = model_profiles
        self._worker_provider = worker_provider or InProcessWorkerProvider()
        self._pending: set[SpecId] = set()

    def begin(
        self,
        *,
        run_id: RunId,
        task_spec: TaskSpec,
        capability_plan: CapabilityPlan,
        version: int = 1,
        continuable: bool = False,
        parent_spec_id: SpecId | None = None,
    ) -> AgentExecutionSpec:
        routing = self._router.route(task_spec.model_requirement, self._profiles)
        spec = AgentExecutionSpec(
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
            continuable=continuable,
            parent_spec_id=parent_spec_id,
        )
        self._pending.add(spec.spec_id)
        return spec

    def commit(self, spec: AgentExecutionSpec) -> AgentExecutionSpec:
        """发布 spec: 从 pending 移除, 交给 worker provider 注册。"""
        self._pending.discard(spec.spec_id)
        self._worker_provider.register(spec)
        return spec

    def rollback(self, spec: AgentExecutionSpec) -> None:
        """回滚未发布 spec: 清理 pending 与 provider 侧注册。"""
        self._pending.discard(spec.spec_id)
        self._worker_provider.unregister(spec.spec_id)

    def create(
        self,
        *,
        run_id: RunId,
        task_spec: TaskSpec,
        capability_plan: CapabilityPlan,
        version: int = 1,
        continuable: bool = False,
        parent_spec_id: SpecId | None = None,
    ) -> AgentExecutionSpec:
        spec = self.begin(
            run_id=run_id,
            task_spec=task_spec,
            capability_plan=capability_plan,
            version=version,
            continuable=continuable,
            parent_spec_id=parent_spec_id,
        )
        return self.commit(spec)

    @property
    def worker_provider(self) -> WorkerProvider:
        return self._worker_provider
