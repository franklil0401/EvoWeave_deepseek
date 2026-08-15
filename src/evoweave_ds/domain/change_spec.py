"""Top-level software change request contract."""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from evoweave_ds.domain.base import DomainModel, utc_now
from evoweave_ds.domain.identifiers import RunId, SpecId
from evoweave_ds.domain.validation import validate_repository_path, validate_unique_strings


class ChangeSpec(DomainModel):
    spec_id: SpecId
    run_id: RunId
    objective: str = Field(min_length=1, max_length=10_000)
    repository: str = Field(min_length=1, max_length=2_048)
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    acceptance_criteria: tuple[str, ...] = Field(min_length=1)
    allowed_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("allowed_paths", "forbidden_paths")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "路径范围")

    @field_validator("acceptance_criteria")
    @classmethod
    def validate_acceptance_criteria(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return validate_unique_strings(values, "acceptance_criteria")

    @model_validator(mode="after")
    def validate_scope(self) -> "ChangeSpec":
        overlap = set(self.allowed_paths) & set(self.forbidden_paths)
        if overlap:
            raise ValueError(f"允许和禁止路径不能重叠：{sorted(overlap)}")
        return self
