"""Resolve and create the repository-local runtime directory layout."""

from dataclasses import dataclass
from pathlib import Path

from evoweave_ds.application.configuration import EvoWeaveConfig


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    root: Path
    run_state: Path
    artifacts: Path
    worker_state: Path
    worker_worktrees: Path
    integration_state: Path
    integration_worktrees: Path
    baseline_state: Path
    baseline_worktrees: Path
    orchestration_database: Path
    events: Path
    reports: Path

    @classmethod
    def create(
        cls,
        repository_root: Path | str,
        config: EvoWeaveConfig,
    ) -> "RuntimeLayout":
        repository = Path(repository_root).resolve(strict=True)
        root = (repository / config.runtime_directory).resolve()
        if root.parent != repository:
            raise ValueError("运行目录必须直接位于目标仓库内")
        layout = cls(
            root=root,
            run_state=root / "state" / "runs",
            artifacts=root / "artifacts",
            worker_state=root / "state" / "worker-workspaces",
            worker_worktrees=root / "worktrees" / "workers",
            integration_state=root / "state" / "integration-workspaces",
            integration_worktrees=root / "worktrees" / "integration",
            baseline_state=root / "state" / "baseline-workspaces",
            baseline_worktrees=root / "worktrees" / "baseline",
            orchestration_database=root / "state" / "orchestration.db",
            events=root / "events",
            reports=root / "reports",
        )
        for directory in (
            layout.run_state,
            layout.artifacts,
            layout.worker_state,
            layout.worker_worktrees,
            layout.integration_state,
            layout.integration_worktrees,
            layout.baseline_state,
            layout.baseline_worktrees,
            layout.events,
            layout.reports,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return layout
