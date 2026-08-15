"""Shared deterministic fixtures for stage 0 contracts."""

import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evoweave_ds.domain.enums import (
    InputModality,
    ModelAvailability,
    ModelTier,
    TaskDifficulty,
)
from evoweave_ds.domain.identifiers import TaskId
from evoweave_ds.domain.model_routing import (
    DifficultyAssessment,
    ModelProfile,
    ModelRequirement,
)


@pytest.fixture
def committed_repository(tmp_path: Path) -> Callable[[str], Path]:
    """Copy one fixture into a temporary Git repository and commit it."""

    fixture_root = Path(__file__).parent / "fixtures" / "repositories"
    counter = 0

    def create(name: str) -> Path:
        nonlocal counter
        counter += 1
        destination = tmp_path / f"{name}-{counter}"
        shutil.copytree(
            fixture_root / name,
            destination,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                ".ruff_cache",
            ),
        )
        _git(destination, "init", "--initial-branch=main")
        _git(destination, "config", "user.name", "EvoWeave Tests")
        _git(destination, "config", "user.email", "tests@evoweave_ds.local")
        _git(destination, "add", "-A")
        _git(destination, "commit", "-m", "fixture")
        return destination

    return create


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def task_id() -> TaskId:
    return TaskId.new()


@pytest.fixture
def low_difficulty() -> DifficultyAssessment:
    return DifficultyAssessment(difficulty=TaskDifficulty.LOW, rationale="单文件确定性修改")


@pytest.fixture
def text_requirement(task_id: TaskId) -> ModelRequirement:
    return ModelRequirement(
        requirement_id="spec_requirement1",
        task_id=task_id,
        difficulty=TaskDifficulty.LOW,
        min_context_tokens=8_000,
        min_output_tokens=1_000,
    )


@pytest.fixture
def text_profile() -> ModelProfile:
    return ModelProfile(
        provider="fake",
        model_id="text-small",
        tier=ModelTier.LOW,
        availability=ModelAvailability.AVAILABLE,
        input_modalities=(InputModality.TEXT,),
        context_window_tokens=32_000,
        max_output_tokens=4_000,
        supports_structured_output=True,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
