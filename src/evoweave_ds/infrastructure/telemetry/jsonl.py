"""Append-only per-run JSONL event recorder with recovered sequence numbers."""

import os
from pathlib import Path
from threading import RLock

from pydantic import JsonValue

from evoweave_ds.domain.enums import EventType
from evoweave_ds.domain.events import DomainEvent
from evoweave_ds.domain.identifiers import EventId, RunId, TaskId


class JsonlEventRecorder:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def record(
        self,
        *,
        run_id: RunId,
        event_type: EventType,
        payload: dict[str, JsonValue],
        task_id: TaskId | None = None,
    ) -> DomainEvent:
        with self._lock:
            sequence = len(self.events_for(run_id)) + 1
            event = DomainEvent(
                event_id=EventId.new(),
                run_id=run_id,
                task_id=task_id,
                sequence=sequence,
                event_type=event_type,
                payload=payload,
            )
            path = self._path_for(run_id)
            with path.open("ab") as stream:
                stream.write(event.model_dump_json().encode("utf-8") + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            return event

    def events_for(self, run_id: RunId) -> tuple[DomainEvent, ...]:
        path = self._path_for(run_id)
        if not path.exists():
            return ()
        return tuple(
            DomainEvent.model_validate_json(line) for line in path.read_bytes().splitlines() if line
        )

    def _path_for(self, run_id: RunId) -> Path:
        path = (self._root / f"{run_id}.jsonl").resolve()
        if path.parent != self._root:
            raise ValueError("事件日志路径越界")
        return path
