from collections.abc import Callable
from pathlib import Path

import pytest

from evoweave_ds.domain.enums import ArtifactKind
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.repository_models import RepositoryAnalysisPolicy
from evoweave_ds.infrastructure.artifacts.memory import InMemoryArtifactStore
from evoweave_ds.repository.evidence_builder import EvidenceBuilder
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.repository.profile_builder import RepositoryProfiler, persist_repository_profile
from evoweave_ds.repository.profile_cache import RepositoryProfileCache, serialize_profile


def test_git_inspector_reads_fixed_commit_and_reports_dirty_worktree(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("single_module")
    inspector = GitInspector(repository)
    committed = inspector.read_bytes("calculator.py")
    (repository / "calculator.py").write_text("uncommitted = True\n", encoding="utf-8")

    assert inspector.state().is_dirty is True
    assert inspector.state().changed_paths == ("calculator.py",)
    assert inspector.read_bytes("calculator.py") == committed
    assert b"uncommitted" not in committed


def test_non_git_directory_has_stable_error_code(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as error:
        GitInspector(tmp_path)
    assert error.value.code is ErrorCode.NOT_GIT_REPOSITORY


def test_git_object_reader_rejects_path_traversal(
    committed_repository: Callable[[str], Path],
) -> None:
    inspector = GitInspector(committed_repository("single_module"))
    with pytest.raises(ValueError, match="仓库路径"):
        inspector.read_bytes("../outside.txt")


def test_repository_analysis_enforces_file_count_limit(
    committed_repository: Callable[[str], Path],
) -> None:
    inspector = GitInspector(committed_repository("single_module"))
    profiler = RepositoryProfiler(policy=RepositoryAnalysisPolicy(max_files=1))

    with pytest.raises(DomainError) as error:
        profiler.build(inspector)
    assert error.value.code is ErrorCode.REPOSITORY_LIMIT_EXCEEDED


def test_profile_is_deterministic_and_cacheable(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("multi_module")
    inspector = GitInspector(repository)
    cache = RepositoryProfileCache()

    first = RepositoryProfiler().build(inspector)
    (repository / "src" / "shop" / "pricing.py").write_text(
        "def changed_only_in_worktree():\n    pass\n", encoding="utf-8"
    )
    second = RepositoryProfiler().build(GitInspector(repository, first.base_commit))
    cache.put(repository_root=str(inspector.repository_root), profile=first)
    cached = cache.get(
        repository_root=str(inspector.repository_root),
        base_commit=first.base_commit,
        analyzer_version=first.analyzer_version,
    )

    assert second == first
    assert second.profile_digest == first.profile_digest
    assert second.model_dump_json() == first.model_dump_json()
    assert cached == first
    assert "changed_only_in_worktree" not in {symbol.name for symbol in second.symbols}


def test_profile_extracts_symbols_dependencies_categories_and_parse_issues(
    committed_repository: Callable[[str], Path],
) -> None:
    profile = RepositoryProfiler().build(GitInspector(committed_repository("multi_module")))

    assert any(
        symbol.qualified_name == "shop.pricing.calculate_discount" for symbol in profile.symbols
    )
    assert any(
        edge.importer_module == "shop.service" and edge.imported_module == "shop.pricing"
        for edge in profile.dependencies
    )
    assert any(file.kind == "test" for file in profile.files)
    assert any(file.kind == "ci" for file in profile.files)
    assert [issue.path for issue in profile.parse_issues] == ["src/shop/legacy_broken.py"]
    assert {command.command_id for command in profile.validation_commands} == {
        "pytest",
        "ruff",
        "typecheck",
    }


def test_evidence_relocates_to_same_lines_and_commit(
    committed_repository: Callable[[str], Path],
) -> None:
    inspector = GitInspector(committed_repository("single_module"))
    profile = RepositoryProfiler().build(inspector)
    symbol = next(item for item in profile.symbols if item.name == "calculate_discount")
    evidence = next(item for item in profile.evidence if item.evidence_id == symbol.evidence_id)

    assert EvidenceBuilder().verify(evidence, inspector) is True


def test_profile_can_be_persisted_as_small_artifact_reference(
    committed_repository: Callable[[str], Path],
) -> None:
    profile = RepositoryProfiler().build(GitInspector(committed_repository("single_module")))
    store = InMemoryArtifactStore()
    reference = persist_repository_profile(profile, store)

    assert reference.kind is ArtifactKind.REPOSITORY_PROFILE
    restored = type(profile).model_validate_json(store.get_bytes(reference.artifact_id))
    assert restored == profile


def test_cache_rejects_tampered_profile(
    committed_repository: Callable[[str], Path],
) -> None:
    inspector = GitInspector(committed_repository("single_module"))
    cache = RepositoryProfileCache()
    profile = RepositoryProfiler(cache=cache).build(inspector)
    tampered = profile.model_copy(update={"profile_digest": "0" * 64})

    with pytest.raises(DomainError) as error:
        serialize_profile(tampered)
    assert error.value.code is ErrorCode.PROFILE_INTEGRITY_ERROR
