"""Append-only domain event contract used by replayable orchestration."""

from datetime import datetime

from pydantic import Field, JsonValue

from evoweave_ds.domain.base import DomainModel, utc_now
from evoweave_ds.domain.enums import EventType
from evoweave_ds.domain.identifiers import EventId, RunId, TaskId


class DomainEvent(DomainModel):
    event_id: EventId
    run_id: RunId
    sequence: int = Field(ge=1)
    event_type: EventType
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    task_id: TaskId | None = None
    schema_version: int = Field(default=1, ge=1)
    occurred_at: datetime = Field(default_factory=utc_now)
