"""Stable CLI output envelope."""

from typing import Any

from pydantic import Field

from evoweave_ds.domain.base import DomainModel


class CliError(DomainModel):
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2_000)


class CliEnvelope(DomainModel):
    ok: bool
    data: dict[str, Any] | None = None
    error: CliError | None = None
