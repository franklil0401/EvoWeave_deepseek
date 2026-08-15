"""Idempotent SQLite graph, decision-ledger, and checkpoint storage."""

import json
from threading import RLock

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.graph_models import GraphSnapshot
from evoweave_ds.domain.identifiers import RunId, SpecId
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.infrastructure.persistence.sqlite import SQLiteDatabase


class SQLiteOrchestrationStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database
        self._lock = RLock()
        self._initialize()

    def save_graph(
        self,
        snapshot: GraphSnapshot,
        task_specs: tuple[TaskSpec, ...],
    ) -> None:
        snapshot_json = snapshot.model_dump_json()
        specs_json = json.dumps(
            [spec.model_dump(mode="json") for spec in task_specs],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock, self._database.connect() as connection:
            existing = connection.execute(
                "SELECT snapshot_json, specs_json FROM graph_snapshots "
                "WHERE run_id = ? AND version = ?",
                (str(snapshot.run_id), snapshot.version),
            ).fetchone()
            if existing is not None:
                if existing != (snapshot_json, specs_json):
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_ERROR,
                        "同一图版本不能写入不同内容",
                    )
                return
            connection.execute(
                "INSERT INTO graph_snapshots "
                "(run_id, graph_id, version, snapshot_json, specs_json) VALUES (?, ?, ?, ?, ?)",
                (
                    str(snapshot.run_id),
                    str(snapshot.graph_id),
                    snapshot.version,
                    snapshot_json,
                    specs_json,
                ),
            )

    def load_latest_graph(
        self,
        run_id: RunId,
    ) -> tuple[GraphSnapshot, tuple[TaskSpec, ...]] | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json, specs_json FROM graph_snapshots "
                "WHERE run_id = ? ORDER BY version DESC LIMIT 1",
                (str(run_id),),
            ).fetchone()
        if row is None:
            return None
        snapshot = GraphSnapshot.model_validate_json(row[0])
        raw_specs = json.loads(row[1])
        specs = tuple(TaskSpec.model_validate(item) for item in raw_specs)
        return snapshot, specs

    def record_decision(
        self,
        *,
        decision_id: SpecId,
        run_id: RunId,
        graph_version: int,
        payload: bytes,
    ) -> bool:
        with self._lock, self._database.connect() as connection:
            existing = connection.execute(
                "SELECT payload FROM decision_ledger WHERE decision_id = ?",
                (str(decision_id),),
            ).fetchone()
            if existing is not None:
                if bytes(existing[0]) != payload:
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_ERROR,
                        "同一调度决策 ID 不能对应不同内容",
                    )
                return False
            connection.execute(
                "INSERT INTO decision_ledger "
                "(decision_id, run_id, graph_version, payload) VALUES (?, ?, ?, ?)",
                (str(decision_id), str(run_id), graph_version, payload),
            )
        return True

    def has_decision(self, decision_id: SpecId) -> bool:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM decision_ledger WHERE decision_id = ?",
                (str(decision_id),),
            ).fetchone()
        return row is not None

    def get_decision_payload(self, decision_id: SpecId) -> bytes | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM decision_ledger WHERE decision_id = ?",
                (str(decision_id),),
            ).fetchone()
        return bytes(row[0]) if row is not None else None

    def save_checkpoint(self, *, run_id: RunId, version: int, payload: bytes) -> None:
        with self._lock, self._database.connect() as connection:
            existing = connection.execute(
                "SELECT payload FROM checkpoints WHERE run_id = ? AND version = ?",
                (str(run_id), version),
            ).fetchone()
            if existing is not None:
                if bytes(existing[0]) != payload:
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_ERROR,
                        "同一检查点版本不能写入不同内容",
                    )
                return
            connection.execute(
                "INSERT INTO checkpoints (run_id, version, payload) VALUES (?, ?, ?)",
                (str(run_id), version, payload),
            )

    def load_latest_checkpoint(self, run_id: RunId) -> bytes | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM checkpoints WHERE run_id = ? ORDER BY version DESC LIMIT 1",
                (str(run_id),),
            ).fetchone()
        return bytes(row[0]) if row is not None else None

    def _initialize(self) -> None:
        with self._database.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_snapshots (
                    run_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    specs_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, version)
                );
                CREATE TABLE IF NOT EXISTS decision_ledger (
                    decision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    graph_version INTEGER NOT NULL,
                    payload BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    PRIMARY KEY (run_id, version)
                );
                """
            )
