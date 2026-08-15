"""Structured result returned by every temporary worker."""

from pydantic import Field, JsonValue, model_validator

from evoweave_ds.domain.artifacts import ArtifactRef, EvidenceRef
from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import ResultStatus, RiskLevel
from evoweave_ds.domain.errors import ErrorCode
from evoweave_ds.domain.identifiers import AgentId, SpecId, TaskId
from evoweave_ds.domain.resources import ResourceUsage


class TaskFailure(DomainModel):
    code: ErrorCode
    message: str = Field(min_length=1, max_length=2_000)
    retryable: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TaskResult(DomainModel):
    task_id: TaskId
    agent_id: AgentId
    execution_spec_id: SpecId
    execution_spec_version: int = Field(ge=1)
    status: ResultStatus
    summary: str = Field(min_length=1, max_length=10_000)
    evidence: tuple[EvidenceRef, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    risk_level: RiskLevel = RiskLevel.LOW
    risk_notes: tuple[str, ...] = ()
    usage: ResourceUsage = Field(default_factory=ResourceUsage)
    failure: TaskFailure | None = None

    @model_validator(mode="after")
    def validate_result_state(self) -> "TaskResult":
        if self.status is ResultStatus.SUCCEEDED:
            if self.failure is not None:
                raise ValueError("成功结果不能包含 failure")
            if not self.evidence:
                raise ValueError("成功结果必须包含至少一条证据")
        elif self.failure is None:
            raise ValueError("非成功结果必须包含 failure")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence 不能重复")
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifacts 不能重复")
        return self
