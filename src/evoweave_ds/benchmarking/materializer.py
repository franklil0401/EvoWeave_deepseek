"""Create deterministic Git repositories from locked local benchmark fixtures."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from evoweave_ds.benchmarking.models import BenchmarkTask

_IGNORED_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".runtime"}
)
_FIXED_DATE = "2026-01-01T00:00:00Z"


@dataclass(frozen=True, slots=True)
class MaterializedRepository:
    path: Path
    base_commit: str
    fixture_sha256: str


class FixtureMaterializer:
    def __init__(self, project_root: Path | str) -> None:
        self._project_root = Path(project_root).resolve(strict=True)
        self._fixture_root = (self._project_root / "tests/fixtures/repositories").resolve(
            strict=True
        )

    def materialize(
        self,
        task: BenchmarkTask,
        destination: Path | str,
    ) -> MaterializedRepository:
        if task.external_repository is not None:
            return self._materialize_external(task)

        source = (self._fixture_root / task.repository_fixture).resolve(strict=True)
        if source.parent != self._fixture_root:
            raise ValueError("benchmark fixture 路径越界")
        fixture_digest = fixture_sha256(source)
        if fixture_digest != task.fixture_sha256:
            raise ValueError("benchmark fixture 内容摘要漂移")
        target = Path(destination).resolve()
        if target.exists():
            raise ValueError("benchmark 目标目录必须不存在")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, ignore=_ignore_generated)
        _git(target, "init", "--initial-branch=main")
        _git(target, "config", "user.name", "EvoWeave Benchmark")
        _git(target, "config", "user.email", "benchmark@evoweave_ds.local")
        _git(target, "config", "core.autocrlf", "false")
        _git(target, "config", "core.filemode", "false")
        _git(target, "add", "-A")
        _git(target, "commit", "--no-gpg-sign", "-m", "EvoWeave benchmark fixture v1")
        commit = _git(target, "rev-parse", "HEAD")
        if commit != task.base_commit:
            raise ValueError("benchmark 固定 commit 漂移")
        return MaterializedRepository(
            path=target,
            base_commit=commit,
            fixture_sha256=fixture_digest,
        )

    def _materialize_external(self, task: BenchmarkTask) -> MaterializedRepository:
        """真实仓库模式: 复用已克隆的外部仓库, 校验固定 commit."""

        if task.external_repository is None:
            raise ValueError("外部仓库模式缺少仓库路径")
        repository = Path(task.external_repository).resolve(strict=True)
        commit = _git(repository, "rev-parse", "HEAD")
        if commit != task.base_commit:
            raise ValueError(f"外部仓库 HEAD 漂移: 期望 {task.base_commit}, 实际 {commit}")
        return MaterializedRepository(
            path=repository,
            base_commit=commit,
            fixture_sha256=task.fixture_sha256,
        )


def fixture_sha256(source: Path | str) -> str:
    root = Path(source).resolve(strict=True)
    digest = sha256()
    for path in sorted(
        (
            item
            for item in root.rglob("*")
            if item.is_file() and not _is_generated(item.relative_to(root))
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        digest.update(len(relative.encode("utf-8")).to_bytes(4, "big"))
        digest.update(relative.encode("utf-8"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _ignore_generated(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _IGNORED_NAMES or name.endswith((".pyc", ".pyo"))}


def _is_generated(path: Path) -> bool:
    return any(part in _IGNORED_NAMES for part in path.parts) or path.name.endswith(
        (".pyc", ".pyo")
    )


def _git(repository: Path, *args: str) -> str:
    environment = {
        **os.environ,
        "GIT_AUTHOR_DATE": _FIXED_DATE,
        "GIT_COMMITTER_DATE": _FIXED_DATE,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    completed = subprocess.run(
        ("git", "-C", str(repository), *args),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    if completed.returncode != 0:
        raise ValueError(f"benchmark Git 操作失败：git {' '.join(args[:2])}")
    return completed.stdout.strip()
