"""Convert CLI inputs into one immutable ChangeSpec."""

from pathlib import Path

from evoweave_ds.domain.change_spec import ChangeSpec
from evoweave_ds.domain.identifiers import RunId, SpecId
from evoweave_ds.repository.git_inspector import GitInspector


class IntakeService:
    def create(
        self,
        *,
        repository: Path | str,
        objective: str,
        acceptance_criteria: tuple[str, ...],
        allowed_paths: tuple[str, ...] = (),
        forbidden_paths: tuple[str, ...] = (),
    ) -> ChangeSpec:
        inspector = GitInspector(repository)
        run_id = RunId.new()
        return ChangeSpec(
            spec_id=SpecId.new(),
            run_id=run_id,
            objective=objective,
            repository=str(inspector.repository_root),
            base_commit=inspector.base_commit,
            acceptance_criteria=acceptance_criteria,
            allowed_paths=allowed_paths,
            forbidden_paths=forbidden_paths,
        )
