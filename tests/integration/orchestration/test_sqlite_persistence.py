from pathlib import Path

import pytest

from evoweave_ds.domain.enums import InputModality, TaskDifficulty
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import DifficultyAssessment, ModelRequirement
from evoweave_ds.domain.ports import CheckpointStore, DecisionLedger, GraphStateStore
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.infrastructure.persistence.graph_repository import SQLiteOrchestrationStore
from evoweave_ds.infrastructure.persistence.sqlite import SQLiteDatabase
from evoweave_ds.orchestration.task_graph import TaskGraph


def test_sqlite_store_round_trips_all_graph_versions_and_is_idempotent(
    tmp_path: Path,
) -> None:
    store = SQLiteOrchestrationStore(SQLiteDatabase(tmp_path / "runtime.db"))
    assert isinstance(store, GraphStateStore)
    assert isinstance(store, DecisionLedger)
    assert isinstance(store, CheckpointStore)
    run_id = RunId.new()
    task = _task()
    graph = TaskGraph.create(run_id=run_id, task_specs=(task,))

    for snapshot, specs in graph.version_records:
        store.save_graph(snapshot, specs)
        store.save_graph(snapshot, specs)
    restored = store.load_latest_graph(run_id)

    assert restored is not None
    assert restored[0] == graph.snapshot
    assert restored[1] == graph.task_specs


def test_decision_ledger_rejects_same_id_with_different_payload(tmp_path: Path) -> None:
    store = SQLiteOrchestrationStore(SQLiteDatabase(tmp_path / "runtime.db"))
    decision_id = SpecId.new()
    run_id = RunId.new()

    assert store.record_decision(
        decision_id=decision_id,
        run_id=run_id,
        graph_version=1,
        payload=b"first",
    )
    assert not store.record_decision(
        decision_id=decision_id,
        run_id=run_id,
        graph_version=1,
        payload=b"first",
    )
    with pytest.raises(DomainError) as error:
        store.record_decision(
            decision_id=decision_id,
            run_id=run_id,
            graph_version=1,
            payload=b"different",
        )
    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_ERROR


def _task() -> TaskSpec:
    task_id = TaskId.new()
    return TaskSpec(
        spec_id=SpecId.new(),
        task_id=task_id,
        change_spec_id=SpecId.new(),
        goal="探索",
        base_commit="a" * 40,
        acceptance_criteria=("完成",),
        read_scope=("src",),
        difficulty=DifficultyAssessment(difficulty=TaskDifficulty.LOW, rationale="测试"),
        model_requirement=ModelRequirement(
            requirement_id=SpecId.new(),
            task_id=task_id,
            difficulty=TaskDifficulty.LOW,
            required_modalities=(InputModality.TEXT,),
        ),
    )
