from evoweave_ds.domain.repository_models import ModuleDependency, ValidationCommand
from evoweave_ds.repository.baseline_runner import (
    BaselineExecution,
    BaselineRunner,
    ScriptedBaselineExecutor,
    existing_failure_ids,
)
from evoweave_ds.repository.dependency_graph import dependency_neighbors
from evoweave_ds.repository.evidence_builder import deterministic_evidence_id
from evoweave_ds.repository.impact_analysis import RequirementClueExtractor


def test_requirement_clues_extract_paths_and_symbols_without_duplicates() -> None:
    clues = RequirementClueExtractor().extract(
        "修改 src/shop/pricing.py 的 calculate_discount，calculate_discount 必须稳定"
    )

    assert clues.paths == ("src/shop/pricing.py",)
    assert "calculate_discount" in clues.symbols
    assert len(clues.terms) == len(set(clues.terms))


def test_requirement_clues_extract_root_level_filename() -> None:
    clues = RequirementClueExtractor().extract("只修改 calculator.py 并保持测试通过")

    assert clues.paths == ("calculator.py",)


def test_baseline_runner_keeps_preexisting_failure_identity() -> None:
    command = ValidationCommand(
        command_id="pytest",
        argv=("python", "-m", "pytest"),
        source="pytest",
    )
    executor = ScriptedBaselineExecutor(
        {"pytest": BaselineExecution(exit_code=1, stdout="one existing failure")}
    )
    results = BaselineRunner(executor).run(base_commit="a" * 40, commands=(command,))

    assert existing_failure_ids(results) == frozenset({"pytest"})
    assert str(results[0].evidence_id).startswith("evidence_")
    assert executor.calls == [("a" * 40, "pytest")]


def test_dependency_neighbors_are_bounded_by_depth() -> None:
    edges = (
        _edge("app.api", "app.service", 1),
        _edge("app.service", "app.store", 2),
    )

    assert dependency_neighbors({"app.api"}, edges, max_depth=1) == {
        "app.api": 0,
        "app.service": 1,
    }


def _edge(importer: str, imported: str, line: int) -> ModuleDependency:
    return ModuleDependency(
        importer_module=importer,
        imported_module=imported,
        path="src/app.py",
        line=line,
        evidence_id=deterministic_evidence_id(importer, imported, line),
    )
