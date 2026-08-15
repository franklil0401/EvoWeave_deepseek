"""Classify pre-existing, new, resolved, and unstable failures."""

from evoweave_ds.domain.enums import FailureClassification, ValidationPhase
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import SpecId
from evoweave_ds.domain.integration_models import FailureDelta, ValidationObservation


class BaselineComparator:
    def compare(
        self,
        observations: tuple[ValidationObservation, ...],
    ) -> tuple[FailureDelta, ...]:
        by_command: dict[SpecId, dict[ValidationPhase, ValidationObservation]] = {}
        for observation in observations:
            phases = by_command.setdefault(observation.command_id, {})
            if observation.phase in phases:
                raise DomainError(ErrorCode.INVALID_SPEC, "同一命令阶段存在重复验证观察")
            phases[observation.phase] = observation

        deltas: list[FailureDelta] = []
        for command_id in sorted(by_command, key=str):
            phases = by_command[command_id]
            if ValidationPhase.BASELINE not in phases or ValidationPhase.CANDIDATE not in phases:
                raise DomainError(ErrorCode.INVALID_SPEC, "每个验证命令都需要基线和候选观察")
            baseline = set(phases[ValidationPhase.BASELINE].failure_keys)
            candidate = set(phases[ValidationPhase.CANDIDATE].failure_keys)
            retry_observation = phases.get(ValidationPhase.CANDIDATE_RETRY)
            retry = set(retry_observation.failure_keys) if retry_observation is not None else None
            unstable = candidate.symmetric_difference(retry) if retry is not None else set()
            stable_candidate = candidate.intersection(retry) if retry is not None else candidate
            deltas.extend(
                FailureDelta(
                    command_id=command_id,
                    failure_key=key,
                    classification=FailureClassification.UNSTABLE,
                )
                for key in sorted(unstable)
            )
            deltas.extend(
                FailureDelta(
                    command_id=command_id,
                    failure_key=key,
                    classification=FailureClassification.PRE_EXISTING,
                )
                for key in sorted(baseline.intersection(stable_candidate))
            )
            deltas.extend(
                FailureDelta(
                    command_id=command_id,
                    failure_key=key,
                    classification=FailureClassification.NEW,
                )
                for key in sorted(stable_candidate.difference(baseline))
            )
            deltas.extend(
                FailureDelta(
                    command_id=command_id,
                    failure_key=key,
                    classification=FailureClassification.RESOLVED,
                )
                for key in sorted(baseline.difference(stable_candidate).difference(unstable))
            )
        return tuple(deltas)
