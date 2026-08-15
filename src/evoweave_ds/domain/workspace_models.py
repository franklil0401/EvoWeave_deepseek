"""Versioned contracts for isolated workspace leases and sandbox execution."""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from evoweave_ds.domain.base import DomainModel, utc_now
from evoweave_ds.domain.enums import WorkspaceAccessMode, WorkspaceLeaseStatus
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId, WorkspaceId
from evoweave_ds.domain.validation import (
    validate_repository_path,
    validate_scope_subset,
    validate_unique_strings,
)


class WorkspaceLease(DomainModel):
    workspace_id: WorkspaceId
    run_id: RunId
    task_id: TaskId
    agent_id: AgentId
    execution_spec_id: SpecId
    execution_spec_version: int = Field(ge=1)
    repository_root: str = Field(min_length=1, max_length=4_096)
    worktree_path: str = Field(min_length=1, max_length=4_096)
    branch_name: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9._/-]+$",
    )
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    access_mode: WorkspaceAccessMode
    read_scope: tuple[str, ...]
    write_scope: tuple[str, ...] = ()
    status: WorkspaceLeaseStatus = WorkspaceLeaseStatus.CREATING
    failure_reason: str | None = Field(default=None, min_length=1, max_length=2_000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    version: int = Field(default=1, ge=1)

    @field_validator("read_scope", "write_scope")
    @classmethod
    def validate_scopes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "工作区路径范围")

    @model_validator(mode="after")
    def validate_lease(self) -> "WorkspaceLease":
        validate_scope_subset(
            self.write_scope,
            self.read_scope,
            child_name="write_scope",
            parent_name="read_scope",
        )
        if self.access_mode is WorkspaceAccessMode.READ_ONLY and self.write_scope:
            raise ValueError("只读租约不能包含 write_scope")
        if self.status is WorkspaceLeaseStatus.FAILED and self.failure_reason is None:
            raise ValueError("失败租约必须记录 failure_reason")
        if self.status is not WorkspaceLeaseStatus.FAILED and self.failure_reason is not None:
            raise ValueError("只有失败租约可以包含 failure_reason")
        return self


class SandboxPolicy(DomainModel):
    network_enabled: bool = False
    max_cpus: float = Field(default=1.0, gt=0, le=64)
    max_memory_mb: int = Field(default=1_024, ge=64, le=262_144)
    max_processes: int = Field(default=64, ge=1, le=4_096)
    max_output_bytes: int = Field(default=1_000_000, ge=1)
    max_timeout_seconds: int = Field(default=1_800, ge=1, le=86_400)

    @model_validator(mode="after")
    def enforce_first_version_network_policy(self) -> "SandboxPolicy":
        if self.network_enabled:
            raise ValueError("第一版沙箱策略禁止开放网络")
        return self
