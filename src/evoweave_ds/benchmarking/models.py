"""Strong benchmark task, run, and aggregate contracts."""

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import InputModality, TaskDifficulty
from evoweave_ds.domain.validation import validate_repository_path, validate_unique_strings


class AgentStrategy(StrEnum):
    SINGLE = "single_agent"
    FIXED_MULTI = "fixed_multi_agent"
    ADAPTIVE = "adaptive_agent"


class ModelStrategy(StrEnum):
    FIXED_LOW = "fixed_low"
    FIXED_HIGH = "fixed_high"
    ADAPTIVE = "adaptive_model"


class BenchmarkRunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BaselineStatus(StrEnum):
    PASSING = "passing"
    FAILING = "failing"


class EvidenceLevel(StrEnum):
    OFFLINE_REPLAY = "offline_replay"
    LIVE_MODEL = "live_model"


class BenchmarkTask(DomainModel):
    benchmark_id: str = Field(pattern=r"^bench-[0-9]{2}-[a-z0-9-]+$")
    repository_source: str = Field(min_length=1, max_length=1_024)
    repository_fixture: str = Field(pattern=r"^[a-z0-9_-]+$")
    external_repository: str | None = Field(default=None, max_length=2_048)
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    objective: str = Field(min_length=1, max_length=10_000)
    acceptance_criteria: tuple[str, ...] = Field(min_length=1)
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    forbidden_paths: tuple[str, ...] = ()
    human_difficulty: TaskDifficulty
    required_modalities: tuple[InputModality, ...] = (InputModality.TEXT,)
    gold_paths: tuple[str, ...] = Field(min_length=1)
    validation_argv: tuple[tuple[str, ...], ...] = Field(min_length=1)
    baseline_status: BaselineStatus
    scenario_tags: tuple[str, ...] = Field(min_length=1)

    @field_validator("allowed_paths", "forbidden_paths", "gold_paths")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(validate_repository_path(item) for item in values)
        return validate_unique_strings(normalized, "benchmark paths")

    @field_validator("acceptance_criteria", "scenario_tags")
    @classmethod
    def validate_unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return validate_unique_strings(values, "benchmark values")

    @field_validator("validation_argv")
    @classmethod
    def validate_commands(
        cls,
        values: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str, ...], ...]:
        if any(not argv or any(not item or "\x00" in item for item in argv) for argv in values):
            raise ValueError("benchmark validation argv 无效")
        return values

    @model_validator(mode="after")
    def validate_task(self) -> "BenchmarkTask":
        if set(self.allowed_paths) & set(self.forbidden_paths):
            raise ValueError("benchmark 允许路径与禁止路径不能重叠")
        return self


class BenchmarkSuite(DomainModel):
    suite_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{5,127}$")
    version: int = Field(ge=1)
    hidden_acceptance_source: str = Field(min_length=1, max_length=1_024)
    hidden_acceptance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tasks: tuple[BenchmarkTask, ...] = Field(min_length=8)

    @model_validator(mode="after")
    def validate_coverage(self) -> "BenchmarkSuite":
        identifiers = [task.benchmark_id for task in self.tasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("benchmark_id 不能重复")
        tags = {tag for task in self.tasks for tag in task.scenario_tags}
        required_tags = {
            "single_file",
            "single_module",
            "multi_serial",
            "multi_parallel",
        }
        missing = required_tags.difference(tags)
        if missing:
            raise ValueError(f"benchmark 场景覆盖不足：{sorted(missing)}")
        return self


class BenchmarkRunRecord(DomainModel):
    benchmark_id: str = Field(pattern=r"^bench-[0-9]{2}-[a-z0-9-]+$")
    run_id: str = Field(min_length=1, max_length=128)
    trial_index: int = Field(default=1, ge=1, le=1_000)
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    agent_strategy: AgentStrategy
    model_strategy: ModelStrategy
    evidence_level: EvidenceLevel
    status: BenchmarkRunStatus
    target_tests_passed: bool
    full_regression_passed: bool
    localization_candidates: tuple[str, ...] = ()
    patch_generated: bool
    patch_applied: bool
    patch_authorized: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    agent_count: int = Field(ge=0)
    task_count: int = Field(ge=0)
    invalid_task_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    orchestrator_context_chars: int = Field(ge=0)
    worker_context_chars: int = Field(ge=0)
    initial_route_valid: bool
    route_hard_constraints_satisfied: bool | None = None
    fallback_count: int = Field(ge=0)
    predicted_difficulty: TaskDifficulty
    selected_model_keys: tuple[str, ...] = ()
    evidence_directory: str | None = Field(default=None, min_length=1, max_length=4_096)
    failure_reason: str | None = Field(default=None, max_length=2_000)

    @field_validator("localization_candidates")
    @classmethod
    def validate_candidates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(validate_repository_path(item) for item in values)
        return validate_unique_strings(normalized, "localization_candidates")

    @model_validator(mode="after")
    def validate_record(self) -> "BenchmarkRunRecord":
        if self.invalid_task_count > self.task_count:
            raise ValueError("无效任务数不能超过任务总数")
        if len(set(self.selected_model_keys)) != len(self.selected_model_keys):
            raise ValueError("实际模型 key 不能重复")
        if self.patch_applied and not self.patch_generated:
            raise ValueError("补丁未生成时不能标记为已应用")
        if self.status is BenchmarkRunStatus.PASSED and not (
            self.target_tests_passed
            and self.full_regression_passed
            and self.patch_applied
            and self.patch_authorized
        ):
            raise ValueError("通过记录必须满足测试、补丁应用和授权条件")
        if self.status is BenchmarkRunStatus.FAILED and self.failure_reason is None:
            raise ValueError("失败记录必须提供 failure_reason")
        return self

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens
