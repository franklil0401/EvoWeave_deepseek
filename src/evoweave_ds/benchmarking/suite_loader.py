"""Load and integrity-check a versioned benchmark suite and its local assets."""

import tempfile
from hashlib import sha256
from pathlib import Path

from pydantic import Field

from evoweave_ds.benchmarking.materializer import FixtureMaterializer, fixture_sha256
from evoweave_ds.benchmarking.models import BenchmarkSuite, BenchmarkTask
from evoweave_ds.domain.base import DomainModel


class SuiteValidationReport(DomainModel):
    suite_id: str
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    fixture_count: int = Field(ge=1)
    verified_commits: tuple[str, ...]
    verified_hidden_acceptance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_benchmark_suite(path: Path | str) -> tuple[BenchmarkSuite, str]:
    source = Path(path).resolve(strict=True)
    payload = source.read_bytes()
    suite = BenchmarkSuite.model_validate_json(payload)
    return suite, sha256(payload).hexdigest()


def validate_benchmark_suite(
    project_root: Path | str,
    suite_path: Path | str,
) -> SuiteValidationReport:
    root = Path(project_root).resolve(strict=True)
    suite, suite_digest = load_benchmark_suite(suite_path)
    hidden_root = (root / "benchmarks/任务集/隐藏验收").resolve(strict=True)
    hidden_acceptance = (root / suite.hidden_acceptance_source).resolve(strict=True)
    if hidden_acceptance.parent != hidden_root:
        raise ValueError("benchmark 隐藏验收路径越界")
    hidden_digest = sha256(hidden_acceptance.read_bytes()).hexdigest()
    if hidden_digest != suite.hidden_acceptance_sha256:
        raise ValueError("benchmark 隐藏验收摘要漂移")
    fixture_root = (root / "tests/fixtures/repositories").resolve(strict=True)
    fixture_tasks: dict[str, BenchmarkTask] = {}
    for task in suite.tasks:
        source = (root / task.repository_source).resolve(strict=True)
        expected_source = (fixture_root / task.repository_fixture).resolve(strict=True)
        if source != expected_source or source.parent != fixture_root:
            raise ValueError(f"{task.benchmark_id} 的仓库来源不在固定 fixture 目录")
        if fixture_sha256(source) != task.fixture_sha256:
            raise ValueError(f"{task.benchmark_id} 的 fixture 摘要漂移")
        existing = fixture_tasks.get(task.repository_fixture)
        if existing is not None and (
            existing.fixture_sha256 != task.fixture_sha256
            or existing.base_commit != task.base_commit
        ):
            raise ValueError("同一 fixture 不能绑定不同摘要或 commit")
        fixture_tasks[task.repository_fixture] = task

    verified_commits: list[str] = []
    materializer = FixtureMaterializer(root)
    with tempfile.TemporaryDirectory(prefix="evoweave_ds-benchmark-") as temporary:
        temporary_root = Path(temporary)
        for fixture_name, task in sorted(fixture_tasks.items()):
            materialized = materializer.materialize(
                task,
                temporary_root / fixture_name,
            )
            verified_commits.append(materialized.base_commit)

    return SuiteValidationReport(
        suite_id=suite.suite_id,
        suite_sha256=suite_digest,
        task_count=len(suite.tasks),
        fixture_count=len(fixture_tasks),
        verified_commits=tuple(sorted(verified_commits)),
        verified_hidden_acceptance_sha256=hidden_digest,
    )
