"""Run required validation scopes and decide from evidence, never model claims."""

from pydantic import Field

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import ValidationPhase, ValidationScope
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.integration_models import (
    IntegrationWorkspaceState,
    ValidationCommand,
    ValidationObservation,
    ValidationReport,
)
from evoweave_ds.domain.ports import ArtifactStore, CommandRunner
from evoweave_ds.validation.baseline_compare import BaselineComparator
from evoweave_ds.validation.command_executor import ValidationCommandExecutor
from evoweave_ds.validation.report import ValidationReportBuilder


class ValidationGatePolicy(DomainModel):
    required_scopes: tuple[ValidationScope, ...] = (
        ValidationScope.LOCAL,
        ValidationScope.IMPACT,
        ValidationScope.FULL,
        ValidationScope.LINT,
    )
    retry_failed_candidate_commands: bool = True
    max_commands: int = Field(default=32, ge=1, le=1_000)


class DeterministicValidationGate:
    def __init__(
        self,
        artifact_store: ArtifactStore,
        policy: ValidationGatePolicy | None = None,
    ) -> None:
        self._policy = policy or ValidationGatePolicy()
        self._executor = ValidationCommandExecutor(artifact_store)
        self._comparator = BaselineComparator()
        self._report_builder = ValidationReportBuilder(artifact_store)

    def run(
        self,
        *,
        state: IntegrationWorkspaceState,
        commands: tuple[ValidationCommand, ...],
        baseline_runner: CommandRunner,
        candidate_runner: CommandRunner,
    ) -> ValidationReport:
        self._validate_commands(commands)
        baseline = self._executor.execute(
            commands,
            runner=baseline_runner,
            phase=ValidationPhase.BASELINE,
        )
        candidate = self._executor.execute(
            commands,
            runner=candidate_runner,
            phase=ValidationPhase.CANDIDATE,
        )
        retry: tuple[ValidationObservation, ...] = ()
        if self._policy.retry_failed_candidate_commands:
            failed_ids = {
                item.command_id
                for item in candidate
                if item.exit_code != 0 or item.timed_out or item.output_truncated
            }
            retry_commands = tuple(
                command for command in commands if command.command_id in failed_ids
            )
            if retry_commands:
                retry = self._executor.execute(
                    retry_commands,
                    runner=candidate_runner,
                    phase=ValidationPhase.CANDIDATE_RETRY,
                    attempt=2,
                )
        observations = (*baseline, *candidate, *retry)
        deltas = self._comparator.compare(observations)
        return self._report_builder.build(
            state=state,
            observations=observations,
            failure_deltas=deltas,
        )

    def _validate_commands(self, commands: tuple[ValidationCommand, ...]) -> None:
        if not commands or len(commands) > self._policy.max_commands:
            raise DomainError(ErrorCode.INVALID_SPEC, "验证命令数量无效")
        command_ids = [item.command_id for item in commands]
        if len(set(command_ids)) != len(command_ids):
            raise DomainError(ErrorCode.INVALID_SPEC, "验证命令 ID 不能重复")
        scopes = {item.scope for item in commands}
        missing = set(self._policy.required_scopes).difference(scopes)
        if missing:
            raise DomainError(
                ErrorCode.INVALID_SPEC,
                "验证命令缺少必需门禁范围",
                details={"missing_scopes": sorted(item.value for item in missing)},
            )
