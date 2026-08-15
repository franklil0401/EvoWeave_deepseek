"""Build explicit local, impact, full-regression, and Ruff gate commands."""

from evoweave_ds.domain.enums import ValidationScope
from evoweave_ds.domain.identifiers import SpecId
from evoweave_ds.domain.integration_models import ValidationCommand
from evoweave_ds.domain.validation import validate_repository_path


class PythonValidationPlanBuilder:
    def build(
        self,
        *,
        local_test_paths: tuple[str, ...],
        impacted_test_paths: tuple[str, ...],
        timeout_seconds: int = 300,
    ) -> tuple[ValidationCommand, ...]:
        local = _validated_test_paths(local_test_paths)
        impacted = _validated_test_paths(impacted_test_paths)
        return (
            ValidationCommand(
                command_id=SpecId.new(),
                name="局部测试",
                argv=("python", "-m", "pytest", "-q", *(local or ("tests",))),
                scope=ValidationScope.LOCAL,
                timeout_seconds=timeout_seconds,
            ),
            ValidationCommand(
                command_id=SpecId.new(),
                name="影响测试",
                argv=("python", "-m", "pytest", "-q", *(impacted or ("tests",))),
                scope=ValidationScope.IMPACT,
                timeout_seconds=timeout_seconds,
            ),
            ValidationCommand(
                command_id=SpecId.new(),
                name="全量回归",
                argv=("python", "-m", "pytest", "-q"),
                scope=ValidationScope.FULL,
                timeout_seconds=timeout_seconds,
            ),
            ValidationCommand(
                command_id=SpecId.new(),
                name="Ruff 门禁",
                argv=("python", "-m", "ruff", "check", "."),
                scope=ValidationScope.LINT,
                timeout_seconds=timeout_seconds,
            ),
        )


def _validated_test_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    validated = tuple(validate_repository_path(path) for path in paths)
    if len(set(validated)) != len(validated):
        raise ValueError("测试路径不能重复")
    return validated
