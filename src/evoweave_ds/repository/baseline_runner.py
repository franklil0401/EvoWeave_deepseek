"""Record pre-change validation results through an injected execution boundary."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from evoweave_ds.domain.repository_models import BaselineCheckResult, ValidationCommand
from evoweave_ds.repository.evidence_builder import EvidenceBuilder


@dataclass(frozen=True, slots=True)
class BaselineExecution:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@runtime_checkable
class BaselineCommandExecutor(Protocol):
    """Execute against an isolated snapshot; implementations must not write the source repo."""

    def execute(
        self,
        *,
        base_commit: str,
        command: ValidationCommand,
    ) -> BaselineExecution: ...


class BaselineRunner:
    def __init__(self, executor: BaselineCommandExecutor) -> None:
        self._executor = executor
        self._evidence_builder = EvidenceBuilder()

    def run(
        self,
        *,
        base_commit: str,
        commands: tuple[ValidationCommand, ...],
    ) -> tuple[BaselineCheckResult, ...]:
        results: list[BaselineCheckResult] = []
        for command in commands:
            execution = self._executor.execute(base_commit=base_commit, command=command)
            evidence = self._evidence_builder.command(
                base_commit=base_commit,
                command_id=command.command_id,
                exit_code=execution.exit_code,
                stdout=execution.stdout,
                stderr=execution.stderr,
                timed_out=execution.timed_out,
            )
            results.append(
                BaselineCheckResult(
                    command=command,
                    exit_code=execution.exit_code,
                    stdout=execution.stdout[:20_000],
                    stderr=execution.stderr[:20_000],
                    timed_out=execution.timed_out,
                    evidence_id=evidence.evidence_id,
                )
            )
        return tuple(results)


def existing_failure_ids(results: tuple[BaselineCheckResult, ...]) -> frozenset[str]:
    """Return command IDs that were already failing before any generated change."""

    return frozenset(result.command.command_id for result in results if not result.passed)


class ScriptedBaselineExecutor:
    """Deterministic executor used by offline tests and development fixtures."""

    def __init__(self, results: dict[str, BaselineExecution]) -> None:
        self._results = dict(results)
        self.calls: list[tuple[str, str]] = []

    def execute(
        self,
        *,
        base_commit: str,
        command: ValidationCommand,
    ) -> BaselineExecution:
        self.calls.append((base_commit, command.command_id))
        return self._results[command.command_id]
