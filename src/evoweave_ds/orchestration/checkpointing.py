"""Compact orchestration checkpoints and exact state restoration."""

from hashlib import sha256

from pydantic import Field, model_validator

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.graph_models import GraphSnapshot
from evoweave_ds.domain.identifiers import RunId, SpecId
from evoweave_ds.domain.ports import CheckpointStore
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.orchestration.control_view import ResultControlSummary
from evoweave_ds.orchestration.progress_detector import ProgressState
from evoweave_ds.orchestration.scheduler import AgentAllocationDecision, TaskLease


class ProcessedDecisionRecord(DomainModel):
    decision_id: SpecId
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_payload(cls, decision_id: SpecId, payload: bytes) -> "ProcessedDecisionRecord":
        return cls(decision_id=decision_id, payload_sha256=sha256(payload).hexdigest())


class OrchestrationCheckpoint(DomainModel):
    run_id: RunId
    version: int = Field(ge=1)
    graph: GraphSnapshot
    task_specs: tuple[TaskSpec, ...]
    execution_specs: tuple[AgentExecutionSpec, ...] = ()
    active_leases: tuple[TaskLease, ...] = ()
    allocation_decisions: tuple[AgentAllocationDecision, ...] = ()
    result_summaries: tuple[ResultControlSummary, ...] = ()
    processed_decisions: tuple[ProcessedDecisionRecord, ...] = ()
    progress_state: ProgressState | None = None
    decision_count: int = Field(default=0, ge=0)
    acceptance_satisfied: bool = False
    finished: bool = False
    finish_summary: str | None = Field(default=None, min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "OrchestrationCheckpoint":
        if self.graph.run_id != self.run_id:
            raise ValueError("检查点图必须属于同一 run")
        if self.finished != (self.finish_summary is not None):
            raise ValueError("finished 与 finish_summary 必须一致")
        decision_ids = [item.decision_id for item in self.processed_decisions]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("processed_decisions 不能重复")
        return self


class CheckpointManager:
    def __init__(self, store: CheckpointStore) -> None:
        self._store = store

    def save(self, checkpoint: OrchestrationCheckpoint) -> None:
        self._store.save_checkpoint(
            run_id=checkpoint.run_id,
            version=checkpoint.version,
            payload=checkpoint.model_dump_json().encode("utf-8"),
        )

    def load(self, run_id: RunId) -> OrchestrationCheckpoint | None:
        payload = self._store.load_latest_checkpoint(run_id)
        return OrchestrationCheckpoint.model_validate_json(payload) if payload is not None else None
