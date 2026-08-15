"""Framework-independent ports implemented by infrastructure adapters."""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import JsonValue

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.artifacts import ArtifactRef
from evoweave_ds.domain.enums import ArtifactKind, EventType
from evoweave_ds.domain.events import DomainEvent
from evoweave_ds.domain.graph_models import GraphSnapshot
from evoweave_ds.domain.identifiers import ArtifactId, RunId, SpecId, TaskId, WorkspaceId
from evoweave_ds.domain.model_routing import (
    ModelProfile,
    ModelRequirement,
    ModelRoutingDecision,
)
from evoweave_ds.domain.task_result import TaskResult
from evoweave_ds.domain.task_spec import TaskSpec


@dataclass(frozen=True, slots=True)
class ModelToolContract:
    """One tool contract sent to the model gateway (name uses underscore form)."""

    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """A structured tool call returned by the gateway (name uses underscore form)."""

    name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    model_key: str
    messages: tuple[str, ...]
    max_output_tokens: int
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    tools: tuple[ModelToolContract, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_key or not self.messages:
            raise ValueError("模型请求必须包含模型标识和消息")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens 必须大于零")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    model_key: str
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    tool_calls: tuple[ModelToolCall, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_key:
            raise ValueError("模型响应必须包含模型标识")
        if min(self.input_tokens, self.output_tokens, self.reasoning_tokens) < 0:
            raise ValueError("模型响应 Token 计数不能为负数")


@runtime_checkable
class ModelGateway(Protocol):
    def list_profiles(self) -> tuple[ModelProfile, ...]: ...

    def complete(self, request: ModelRequest) -> ModelResponse: ...


@runtime_checkable
class ModelRouter(Protocol):
    def route(
        self,
        requirement: ModelRequirement,
        profiles: tuple[ModelProfile, ...],
    ) -> ModelRoutingDecision: ...


@runtime_checkable
class WorkerAdapter(Protocol):
    def execute(self, execution_spec: AgentExecutionSpec) -> TaskResult: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        kind: ArtifactKind,
    ) -> ArtifactRef: ...

    def get_bytes(self, artifact_id: ArtifactId) -> bytes: ...

    def get_ref(self, artifact_id: ArtifactId) -> ArtifactRef: ...

    def update_ref(self, ref: ArtifactRef) -> None: ...


@runtime_checkable
class WorkspaceAdapter(Protocol):
    @property
    def workspace_id(self) -> WorkspaceId: ...

    @property
    def task_id(self) -> TaskId: ...

    def read_text(self, path: str) -> str: ...

    def write_text(self, path: str, content: str) -> None: ...

    def list_paths(self, prefix: str | None = None) -> tuple[str, ...]: ...


@runtime_checkable
class WorkspaceProvider(Protocol):
    def for_execution(self, execution_spec: AgentExecutionSpec) -> WorkspaceAdapter: ...


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: int = 0
    output_truncated: bool = False

    def __post_init__(self) -> None:
        if self.duration_ms < 0:
            raise ValueError("duration_ms 不能为负数")


@runtime_checkable
class CommandRunner(Protocol):
    def run(self, argv: tuple[str, ...], *, timeout_seconds: int) -> CommandResult: ...


@runtime_checkable
class EventRecorder(Protocol):
    def record(
        self,
        *,
        run_id: RunId,
        event_type: EventType,
        payload: dict[str, JsonValue],
        task_id: TaskId | None = None,
    ) -> DomainEvent: ...


@runtime_checkable
class GraphStateStore(Protocol):
    def save_graph(
        self,
        snapshot: GraphSnapshot,
        task_specs: tuple[TaskSpec, ...],
    ) -> None: ...

    def load_latest_graph(
        self,
        run_id: RunId,
    ) -> tuple[GraphSnapshot, tuple[TaskSpec, ...]] | None: ...


@runtime_checkable
class DecisionLedger(Protocol):
    def record_decision(
        self,
        *,
        decision_id: SpecId,
        run_id: RunId,
        graph_version: int,
        payload: bytes,
    ) -> bool: ...

    def has_decision(self, decision_id: SpecId) -> bool: ...

    def get_decision_payload(self, decision_id: SpecId) -> bytes | None: ...


@runtime_checkable
class CheckpointStore(Protocol):
    def save_checkpoint(self, *, run_id: RunId, version: int, payload: bytes) -> None: ...

    def load_latest_checkpoint(self, run_id: RunId) -> bytes | None: ...
