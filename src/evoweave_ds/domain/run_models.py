"""Durable top-level run state exposed by the CLI."""

from datetime import datetime

from pydantic import Field, model_validator

from evoweave_ds.domain.base import DomainModel, utc_now
from evoweave_ds.domain.change_spec import ChangeSpec
from evoweave_ds.domain.enums import RunStatus
from evoweave_ds.domain.errors import ErrorCode
from evoweave_ds.domain.identifiers import ArtifactId, RunId


class RunManifest(DomainModel):
    run_id: RunId
    status: RunStatus
    change_spec: ChangeSpec
    repository_profile_artifact_id: ArtifactId | None = None
    final_patch_artifact_id: ArtifactId | None = None
    validation_report_artifact_id: ArtifactId | None = None
    message: str = Field(default="运行已初始化", min_length=1, max_length=10_000)
    error_code: ErrorCode | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> "RunManifest":
        if self.change_spec.run_id != self.run_id:
            raise ValueError("RunManifest 与 ChangeSpec 必须属于同一 run")
        if self.status is RunStatus.FAILED and self.error_code is None:
            raise ValueError("失败运行必须记录 error_code")
        if self.status is not RunStatus.FAILED and self.error_code is not None:
            raise ValueError("非失败运行不能记录 error_code")
        if self.status is RunStatus.ANALYZED and self.repository_profile_artifact_id is None:
            raise ValueError("已分析运行必须引用仓库画像")
        if self.status is RunStatus.COMPLETED and (
            self.final_patch_artifact_id is None or self.validation_report_artifact_id is None
        ):
            raise ValueError("完成运行必须引用最终补丁和验证报告")
        return self
