"""Strong contracts for explainable, non-price-based model routing."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from evoweave_ds.domain.base import DomainModel, utc_now
from evoweave_ds.domain.enums import (
    InputModality,
    ModelAvailability,
    ModelTier,
    TaskDifficulty,
)
from evoweave_ds.domain.identifiers import EvidenceId, SpecId, TaskId


class DifficultyAssessment(DomainModel):
    difficulty: TaskDifficulty
    rationale: str = Field(min_length=1, max_length=2_000)
    evidence_ids: tuple[EvidenceId, ...] = ()
    version: int = Field(default=1, ge=1)


class ModelRequirement(DomainModel):
    requirement_id: SpecId
    task_id: TaskId
    difficulty: TaskDifficulty
    required_modalities: tuple[InputModality, ...] = (InputModality.TEXT,)
    min_context_tokens: int = Field(default=1, ge=1)
    min_output_tokens: int = Field(default=1, ge=1)
    requires_tool_calling: bool = False
    requires_structured_output: bool = False
    requires_thinking: bool = False
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_modalities(self) -> "ModelRequirement":
        if not self.required_modalities:
            raise ValueError("required_modalities 不能为空")
        if len(set(self.required_modalities)) != len(self.required_modalities):
            raise ValueError("required_modalities 不能重复")
        if InputModality.TEXT not in self.required_modalities:
            raise ValueError("第一版任务必须包含 text 输入模态")
        return self


class ModelProfile(DomainModel):
    provider: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$")
    model_id: str = Field(min_length=1, max_length=255)
    tier: ModelTier
    availability: ModelAvailability
    input_modalities: tuple[InputModality, ...]
    context_window_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    supports_tool_calling: bool = False
    supports_structured_output: bool = False
    supports_thinking: bool = False
    stable_priority: int = Field(default=100, ge=0)
    snapshot: str | None = Field(default=None, max_length=255)
    checked_at: datetime | None = None

    @model_validator(mode="after")
    def validate_profile(self) -> "ModelProfile":
        if not self.input_modalities:
            raise ValueError("input_modalities 不能为空")
        if len(set(self.input_modalities)) != len(self.input_modalities):
            raise ValueError("input_modalities 不能重复")
        if InputModality.TEXT not in self.input_modalities:
            raise ValueError("第一版模型必须支持 text 输入")
        if self.availability is ModelAvailability.AVAILABLE and self.checked_at is None:
            raise ValueError("可用模型必须记录 checked_at")
        return self

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model_id}"


class ModelCandidateRejection(DomainModel):
    model_key: str = Field(min_length=3, max_length=320)
    reasons: tuple[str, ...] = Field(min_length=1)


class ModelRoutingDecision(DomainModel):
    decision_id: SpecId
    requirement_id: SpecId
    requirement_version: int = Field(ge=1)
    selected_model_key: str = Field(min_length=3, max_length=320)
    selected_availability: Literal[ModelAvailability.AVAILABLE] = ModelAvailability.AVAILABLE
    selected_snapshot: str | None = Field(default=None, max_length=255)
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    reason: str = Field(min_length=1, max_length=2_000)
    fallback_model_keys: tuple[str, ...] = ()
    rejected_candidates: tuple[ModelCandidateRejection, ...] = ()
    capability_snapshot_at: datetime = Field(default_factory=utc_now)
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_candidates(self) -> "ModelRoutingDecision":
        if self.selected_model_key in self.fallback_model_keys:
            raise ValueError("入选模型不能同时出现在回退列表")
        if len(set(self.fallback_model_keys)) != len(self.fallback_model_keys):
            raise ValueError("回退模型不能重复")
        rejected = {item.model_key for item in self.rejected_candidates}
        if self.selected_model_key in rejected:
            raise ValueError("入选模型不能同时被拒绝")
        return self
