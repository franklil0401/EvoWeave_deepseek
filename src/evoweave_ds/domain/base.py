"""Shared primitives for immutable domain models."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Base class for value-oriented, JSON-serializable domain contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


def utc_now() -> datetime:
    """Return a timezone-aware timestamp suitable for persisted events."""

    return datetime.now(UTC)
