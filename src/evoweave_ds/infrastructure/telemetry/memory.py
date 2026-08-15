"""Append-only in-memory event recorder with per-run sequencing."""

from collections import defaultdict

from pydantic import JsonValue

from evoweave_ds.domain.enums import EventType
from evoweave_ds.domain.events import DomainEvent
from evoweave_ds.domain.identifiers import EventId, RunId, TaskId


class InMemoryEventRecorder:
    def __init__(self) -> None:
        self._events: list[DomainEvent] = []
        self._sequences: dict[RunId, int] = defaultdict(int)

    def record(
        self,
        *,
        run_id: RunId,
        event_type: EventType,
        payload: dict[str, JsonValue],
        task_id: TaskId | None = None,
    ) -> DomainEvent:
        self._sequences[run_id] += 1
        event = DomainEvent(
            event_id=EventId.new(),
            run_id=run_id,
            task_id=task_id,
            sequence=self._sequences[run_id],
            event_type=event_type,
            payload=payload,
        )
        self._events.append(event)
        return event

    def events_for(self, run_id: RunId) -> tuple[DomainEvent, ...]:
        return tuple(event for event in self._events if event.run_id == run_id)
