"""Detect duplicate task proposals and bounded no-progress decision loops."""

from hashlib import sha256

from pydantic import Field

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.graph_models import GraphSnapshot
from evoweave_ds.domain.task_spec import TaskSpec


class ProgressState(DomainModel):
    last_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    unchanged_decisions: int = Field(default=0, ge=0)


class ProgressDetector:
    def graph_fingerprint(self, snapshot: GraphSnapshot) -> str:
        payload = "|".join(
            f"{node.task_id}:{node.status}:{node.task_spec_version}:{node.attempts}"
            for node in sorted(snapshot.nodes, key=lambda item: str(item.task_id))
        )
        return sha256(payload.encode()).hexdigest()

    def observe(
        self,
        snapshot: GraphSnapshot,
        previous: ProgressState | None,
    ) -> ProgressState:
        fingerprint = self.graph_fingerprint(snapshot)
        unchanged = (
            previous.unchanged_decisions + 1
            if previous is not None and previous.last_fingerprint == fingerprint
            else 0
        )
        return ProgressState(last_fingerprint=fingerprint, unchanged_decisions=unchanged)

    def reject_duplicate_specs(
        self,
        incoming: tuple[TaskSpec, ...],
        existing: tuple[TaskSpec, ...],
    ) -> None:
        existing_fingerprints = {_spec_fingerprint(spec) for spec in existing}
        incoming_fingerprints = [_spec_fingerprint(spec) for spec in incoming]
        if len(incoming_fingerprints) != len(set(incoming_fingerprints)) or any(
            fingerprint in existing_fingerprints for fingerprint in incoming_fingerprints
        ):
            raise DomainError(ErrorCode.POLICY_REJECTED, "调度决策包含重复任务")


def _spec_fingerprint(spec: TaskSpec) -> str:
    payload = "|".join(
        (
            spec.goal.casefold(),
            ",".join(sorted(spec.read_scope)),
            ",".join(sorted(spec.write_scope)),
            ",".join(sorted(item.value for item in spec.required_modalities)),
        )
    )
    return sha256(payload.encode()).hexdigest()
