"""Execute bounded validation commands and persist exact command evidence."""

import json
import re

from evoweave_ds.domain.artifacts import ArtifactRef
from evoweave_ds.domain.enums import ArtifactKind, ValidationPhase
from evoweave_ds.domain.integration_models import ValidationCommand, ValidationObservation
from evoweave_ds.domain.ports import ArtifactStore, CommandResult, CommandRunner

_PYTEST_FAILURE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)
_RUFF_FAILURE = re.compile(r"^(.+?):\d+:\d+:\s+([A-Z]+\d+)\b", re.MULTILINE)


class ValidationCommandExecutor:
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    def execute(
        self,
        commands: tuple[ValidationCommand, ...],
        *,
        runner: CommandRunner,
        phase: ValidationPhase,
        attempt: int = 1,
    ) -> tuple[ValidationObservation, ...]:
        observations: list[ValidationObservation] = []
        for command in commands:
            result = runner.run(command.argv, timeout_seconds=command.timeout_seconds)
            log_ref = self._persist_log(command, phase, attempt, result)
            observations.append(
                ValidationObservation(
                    command_id=command.command_id,
                    command_name=command.name,
                    scope=command.scope,
                    phase=phase,
                    attempt=attempt,
                    argv=command.argv,
                    exit_code=result.exit_code,
                    timed_out=result.timed_out,
                    duration_ms=result.duration_ms,
                    output_truncated=result.output_truncated,
                    failure_keys=_failure_keys(command, result),
                    log_ref=log_ref,
                )
            )
        return tuple(observations)

    def _persist_log(
        self,
        command: ValidationCommand,
        phase: ValidationPhase,
        attempt: int,
        result: CommandResult,
    ) -> ArtifactRef:
        payload = json.dumps(
            {
                "command_id": str(command.command_id),
                "name": command.name,
                "scope": command.scope.value,
                "phase": phase.value,
                "attempt": attempt,
                "argv": result.argv,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_ms": result.duration_ms,
                "output_truncated": result.output_truncated,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self._artifact_store.put_bytes(
            payload,
            media_type="application/json",
            kind=ArtifactKind.COMMAND_LOG,
        )


def _failure_keys(command: ValidationCommand, result: CommandResult) -> tuple[str, ...]:
    if result.exit_code == 0 and not result.timed_out:
        return ()
    if result.timed_out:
        return (f"command:{command.command_id}:timeout",)
    combined = f"{result.stdout}\n{result.stderr}"
    keys = {f"pytest:{match.group(1)}" for match in _PYTEST_FAILURE.finditer(combined)}
    keys.update(
        f"ruff:{match.group(1)}:{match.group(2)}" for match in _RUFF_FAILURE.finditer(combined)
    )
    if not keys:
        keys.add(f"command:{command.command_id}:nonzero")
    return tuple(sorted(keys))
