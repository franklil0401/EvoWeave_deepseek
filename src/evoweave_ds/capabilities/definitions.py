"""Strong contracts shared by capability definitions and executions."""

from dataclasses import dataclass
from typing import Protocol

from pydantic import Field, JsonValue

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.artifacts import ArtifactRef, EvidenceRef
from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import CapabilityAccess, RiskLevel
from evoweave_ds.domain.ports import ArtifactStore, CommandRunner, WorkspaceAdapter


class CapabilityDefinition(DomainModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    version: int = Field(default=1, ge=1)
    description: str = Field(min_length=1, max_length=2_000)
    access: CapabilityAccess
    risk_level: RiskLevel
    input_schema: dict[str, JsonValue] = Field(default_factory=dict)


class CapabilityResult(DomainModel):
    summary: str = Field(min_length=1, max_length=2_000)
    details: dict[str, JsonValue] = Field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True, slots=True)
class CapabilityContext:
    execution_spec: AgentExecutionSpec
    workspace: WorkspaceAdapter
    artifact_store: ArtifactStore
    command_runner: CommandRunner | None = None


class Capability(Protocol):
    @property
    def definition(self) -> CapabilityDefinition: ...

    def invoke(
        self,
        arguments: dict[str, JsonValue],
        context: CapabilityContext,
    ) -> CapabilityResult: ...
