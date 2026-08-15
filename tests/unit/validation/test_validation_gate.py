from collections import defaultdict, deque

from evoweave_ds.domain.enums import (
    FailureClassification,
    IntegrationStatus,
    ValidationScope,
)
from evoweave_ds.domain.identifiers import ArtifactId, IntegrationId, RunId, SpecId, TaskId
from evoweave_ds.domain.integration_models import (
    AppliedPatchRecord,
    IntegrationWorkspaceState,
    ValidationCommand,
)
from evoweave_ds.domain.ports import CommandResult
from evoweave_ds.infrastructure.artifacts.memory import InMemoryArtifactStore
from evoweave_ds.validation.gate import DeterministicValidationGate
from evoweave_ds.validation.plan import PythonValidationPlanBuilder


class SequenceCommandRunner:
    def __init__(self, results: dict[tuple[str, ...], tuple[CommandResult, ...]]) -> None:
        self._results = {key: deque(value) for key, value in results.items()}
        self.calls: defaultdict[tuple[str, ...], int] = defaultdict(int)

    def run(self, argv: tuple[str, ...], *, timeout_seconds: int) -> CommandResult:
        assert timeout_seconds > 0
        self.calls[argv] += 1
        return self._results[argv].popleft()


def test_gate_classifies_preexisting_new_and_unstable_failures() -> None:
    commands = _commands()
    baseline = SequenceCommandRunner(
        {
            commands[0].argv: (_failed(commands[0], "FAILED tests/test_old.py::test_old - x"),),
            commands[1].argv: (_passed(commands[1]),),
            commands[2].argv: (_passed(commands[2]),),
            commands[3].argv: (_passed(commands[3]),),
        }
    )
    candidate = SequenceCommandRunner(
        {
            commands[0].argv: (
                _failed(commands[0], "FAILED tests/test_old.py::test_old - x"),
                _failed(commands[0], "FAILED tests/test_old.py::test_old - x"),
            ),
            commands[1].argv: (
                _failed(commands[1], "FAILED tests/test_new.py::test_new - x"),
                _failed(commands[1], "FAILED tests/test_new.py::test_new - x"),
            ),
            commands[2].argv: (
                _failed(commands[2], "FAILED tests/test_flaky.py::test_flaky - x"),
                _passed(commands[2]),
            ),
            commands[3].argv: (_passed(commands[3]),),
        }
    )
    store = InMemoryArtifactStore()

    report = DeterministicValidationGate(store).run(
        state=_state(),
        commands=commands,
        baseline_runner=baseline,
        candidate_runner=candidate,
    )

    classifications = {(item.failure_key, item.classification) for item in report.failure_deltas}
    assert (
        "pytest:tests/test_old.py::test_old",
        FailureClassification.PRE_EXISTING,
    ) in classifications
    assert (
        "pytest:tests/test_new.py::test_new",
        FailureClassification.NEW,
    ) in classifications
    assert (
        "pytest:tests/test_flaky.py::test_flaky",
        FailureClassification.UNSTABLE,
    ) in classifications
    assert report.accepted is False
    assert report.report_ref is not None
    assert report.applied_patches[0].artifact_id
    assert store.get_bytes(report.report_ref.artifact_id)
    assert all(store.get_bytes(item.log_ref.artifact_id) for item in report.observations)


def test_preexisting_failure_alone_does_not_hide_or_create_a_regression() -> None:
    commands = _commands()
    old_failure = "FAILED tests/test_old.py::test_old - known"
    baseline_results = {
        command.argv: (_failed(command, old_failure) if index == 0 else _passed(command),)
        for index, command in enumerate(commands)
    }
    candidate_results = {
        command.argv: (
            (_failed(command, old_failure), _failed(command, old_failure))
            if index == 0
            else (_passed(command),)
        )
        for index, command in enumerate(commands)
    }

    report = DeterministicValidationGate(InMemoryArtifactStore()).run(
        state=_state(),
        commands=commands,
        baseline_runner=SequenceCommandRunner(baseline_results),
        candidate_runner=SequenceCommandRunner(candidate_results),
    )

    assert report.accepted is True
    assert {item.classification for item in report.failure_deltas} == {
        FailureClassification.PRE_EXISTING
    }


def test_preexisting_command_error_is_still_rejected_as_infrastructure_failure() -> None:
    commands = _commands()
    command_error = "test runner could not collect the requested path"
    baseline_results = {
        command.argv: (_failed(command, command_error) if index == 0 else _passed(command),)
        for index, command in enumerate(commands)
    }
    candidate_results = {
        command.argv: (
            (_failed(command, command_error), _failed(command, command_error))
            if index == 0
            else (_passed(command),)
        )
        for index, command in enumerate(commands)
    }

    report = DeterministicValidationGate(InMemoryArtifactStore()).run(
        state=_state(),
        commands=commands,
        baseline_runner=SequenceCommandRunner(baseline_results),
        candidate_runner=SequenceCommandRunner(candidate_results),
    )

    assert report.accepted is False
    assert any(item.failure_key.startswith("command:") for item in report.failure_deltas)


def test_python_plan_contains_local_impact_full_and_ruff_gates() -> None:
    commands = PythonValidationPlanBuilder().build(
        local_test_paths=("tests/test_local.py",),
        impacted_test_paths=("tests/test_impact.py",),
    )

    assert [item.scope for item in commands] == [
        ValidationScope.LOCAL,
        ValidationScope.IMPACT,
        ValidationScope.FULL,
        ValidationScope.LINT,
    ]
    assert commands[0].argv[-1] == "tests/test_local.py"
    assert commands[1].argv[-1] == "tests/test_impact.py"
    assert commands[2].argv == ("python", "-m", "pytest", "-q", "tests")
    assert commands[3].argv == ("python", "-m", "ruff", "check", ".")


def _commands() -> tuple[ValidationCommand, ...]:
    return tuple(
        ValidationCommand(
            command_id=SpecId.new(),
            name=scope.value,
            argv=("pytest", scope.value),
            scope=scope,
        )
        for scope in (
            ValidationScope.LOCAL,
            ValidationScope.IMPACT,
            ValidationScope.FULL,
            ValidationScope.LINT,
        )
    )


def _state() -> IntegrationWorkspaceState:
    base_commit = "a" * 40
    candidate_commit = "b" * 40
    applied = AppliedPatchRecord(
        sequence=1,
        artifact_id=ArtifactId.new(),
        task_id=TaskId.new(),
        before_commit=base_commit,
        after_commit=candidate_commit,
        changed_paths=("src/app.py",),
    )
    return IntegrationWorkspaceState(
        integration_id=IntegrationId.new(),
        run_id=RunId.new(),
        repository_root="C:/repo",
        worktree_path="C:/runtime/integration",
        branch_name="evoweave_ds/integration/test/state",
        base_commit=base_commit,
        head_commit=candidate_commit,
        status=IntegrationStatus.ACTIVE,
        applied_patches=(applied,),
    )


def _passed(command: ValidationCommand) -> CommandResult:
    return CommandResult(argv=command.argv, exit_code=0, stdout="passed", duration_ms=1)


def _failed(command: ValidationCommand, stdout: str) -> CommandResult:
    return CommandResult(argv=command.argv, exit_code=1, stdout=stdout, duration_ms=1)
