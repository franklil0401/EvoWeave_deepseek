"""Immutable contracts for patch integration and deterministic validation."""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from evoweave_ds.domain.artifacts import ArtifactRef, PatchArtifact
from evoweave_ds.domain.base import DomainModel, utc_now
from evoweave_ds.domain.enums import (
    ArtifactKind,
    FailureClassification,
    IntegrationStatus,
    PatchConflictKind,
    ValidationPhase,
    ValidationScope,
)
from evoweave_ds.domain.identifiers import (
    ArtifactId,
    IntegrationId,
    RunId,
    SpecId,
    TaskId,
)
from evoweave_ds.domain.validation import validate_repository_path, validate_unique_strings


class GuardedPatch(DomainModel):
    artifact: PatchArtifact
    parsed_paths: tuple[str, ...] = Field(min_length=1)

    @field_validator("parsed_paths")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "补丁解析路径")

    @model_validator(mode="after")
    def match_claimed_paths(self) -> "GuardedPatch":
        if set(self.parsed_paths) != set(self.artifact.changed_paths):
            raise ValueError("补丁声明路径与实际内容不一致")
        return self


class PatchConflict(DomainModel):
    kind: PatchConflictKind
    message: str = Field(min_length=1, max_length=2_000)
    artifact_ids: tuple[ArtifactId, ...] = Field(min_length=1)
    paths: tuple[str, ...] = ()

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "冲突路径")


class AppliedPatchRecord(DomainModel):
    sequence: int = Field(ge=1)
    artifact_id: ArtifactId
    task_id: TaskId
    before_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    after_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    changed_paths: tuple[str, ...] = Field(min_length=1)
    applied_at: datetime = Field(default_factory=utc_now)

    @field_validator("changed_paths")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "已应用补丁路径")

    @model_validator(mode="after")
    def validate_commits(self) -> "AppliedPatchRecord":
        if self.before_commit == self.after_commit:
            raise ValueError("应用补丁前后 commit 不能相同")
        return self


class IntegrationWorkspaceState(DomainModel):
    integration_id: IntegrationId
    run_id: RunId
    repository_root: str = Field(min_length=1, max_length=4_096)
    worktree_path: str = Field(min_length=1, max_length=4_096)
    branch_name: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._/-]+$")
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    head_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    status: IntegrationStatus = IntegrationStatus.CREATING
    applied_patches: tuple[AppliedPatchRecord, ...] = ()
    failure_reason: str | None = Field(default=None, min_length=1, max_length=2_000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_state(self) -> "IntegrationWorkspaceState":
        if self.status is IntegrationStatus.FAILED and self.failure_reason is None:
            raise ValueError("失败集成工作区必须记录原因")
        if self.status is not IntegrationStatus.FAILED and self.failure_reason is not None:
            raise ValueError("非失败集成工作区不能记录失败原因")
        sequences = [item.sequence for item in self.applied_patches]
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError("补丁应用序号必须从 1 连续递增")
        expected_head = (
            self.applied_patches[-1].after_commit if self.applied_patches else self.base_commit
        )
        if self.head_commit != expected_head:
            raise ValueError("集成状态 HEAD 与补丁记录不一致")
        return self


class IntegratedPatchSet(DomainModel):
    integration_id: IntegrationId
    run_id: RunId
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    candidate_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    ref: ArtifactRef
    changed_paths: tuple[str, ...] = Field(min_length=1)
    source_artifact_ids: tuple[ArtifactId, ...] = Field(min_length=1)

    @field_validator("changed_paths")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "最终补丁路径")

    @model_validator(mode="after")
    def validate_patch_set(self) -> "IntegratedPatchSet":
        if self.ref.kind is not ArtifactKind.PATCH:
            raise ValueError("最终集成产物类型必须为 patch")
        if self.base_commit == self.candidate_commit:
            raise ValueError("最终集成 commit 必须不同于基线")
        if len(set(self.source_artifact_ids)) != len(self.source_artifact_ids):
            raise ValueError("来源补丁不能重复")
        return self


class ValidationCommand(DomainModel):
    command_id: SpecId
    name: str = Field(min_length=1, max_length=255)
    argv: tuple[str, ...] = Field(min_length=1)
    scope: ValidationScope
    timeout_seconds: int = Field(default=300, ge=1, le=86_400)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or "\x00" in value for value in values):
            raise ValueError("验证命令参数不能为空或包含 NUL")
        return values


class ValidationObservation(DomainModel):
    command_id: SpecId
    command_name: str = Field(min_length=1, max_length=255)
    scope: ValidationScope
    phase: ValidationPhase
    attempt: int = Field(default=1, ge=1)
    argv: tuple[str, ...] = Field(min_length=1)
    exit_code: int
    timed_out: bool = False
    duration_ms: int = Field(ge=0)
    output_truncated: bool = False
    failure_keys: tuple[str, ...] = ()
    log_ref: ArtifactRef

    @model_validator(mode="after")
    def validate_log(self) -> "ValidationObservation":
        if self.log_ref.kind is not ArtifactKind.COMMAND_LOG:
            raise ValueError("验证观察必须绑定命令日志产物")
        if self.exit_code == 0 and (self.timed_out or self.failure_keys):
            raise ValueError("成功命令不能包含超时或失败键")
        return self


class FailureDelta(DomainModel):
    command_id: SpecId
    failure_key: str = Field(min_length=1, max_length=2_000)
    classification: FailureClassification


class ValidationReport(DomainModel):
    report_id: SpecId
    run_id: RunId
    integration_id: IntegrationId
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    candidate_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    accepted: bool
    applied_patches: tuple[AppliedPatchRecord, ...] = ()
    observations: tuple[ValidationObservation, ...] = Field(min_length=1)
    failure_deltas: tuple[FailureDelta, ...] = ()
    report_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ValidationReport":
        blocking = {
            FailureClassification.NEW,
            FailureClassification.UNSTABLE,
        }
        expected_acceptance = not any(
            item.classification in blocking for item in self.failure_deltas
        ) and not any(
            item.timed_out
            or item.output_truncated
            or any(key.startswith("command:") for key in item.failure_keys)
            for item in self.observations
        )
        if self.accepted != expected_acceptance:
            raise ValueError("验收结论与失败增量不一致")
        if self.report_ref is not None and self.report_ref.kind is not ArtifactKind.TEST_REPORT:
            raise ValueError("验证报告产物类型必须为 test_report")
        return self
