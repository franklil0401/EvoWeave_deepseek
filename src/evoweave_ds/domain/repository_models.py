"""Immutable contracts for repository snapshots, profiles, and impact evidence."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import TaskDifficulty
from evoweave_ds.domain.identifiers import EvidenceId
from evoweave_ds.domain.model_routing import DifficultyAssessment, ModelRequirement
from evoweave_ds.domain.validation import validate_repository_path, validate_unique_strings

RepositoryFileKind = Literal[
    "python_source",
    "test",
    "configuration",
    "build",
    "ci",
    "documentation",
    "other",
]
PythonSymbolKind = Literal["module", "class", "function", "method"]
ValidationCommandSource = Literal["pytest", "ruff", "project"]


class RepositoryAnalysisPolicy(DomainModel):
    max_files: int = Field(default=10_000, ge=1)
    max_total_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    max_python_file_bytes: int = Field(default=2 * 1024 * 1024, ge=1)


class GitRepositoryState(DomainModel):
    repository_root: str = Field(min_length=1, max_length=4_096)
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    is_dirty: bool
    changed_paths: tuple[str, ...] = ()

    @field_validator("changed_paths")
    @classmethod
    def validate_changed_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_repository_path(value) for value in values)
        return validate_unique_strings(validated, "changed_paths")


class RepositoryBlob(DomainModel):
    path: str
    object_id: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    mode: str = Field(pattern=r"^[0-9]{6}$")
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_path(value)

    @property
    def is_regular_file(self) -> bool:
        return self.mode in {"100644", "100755"}


class RepositoryFile(DomainModel):
    path: str
    object_id: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: RepositoryFileKind
    language: str | None = Field(default=None, max_length=64)
    size_bytes: int = Field(ge=0)
    line_count: int = Field(ge=0)
    module_name: str | None = Field(default=None, min_length=1, max_length=1_024)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_path(value)


class PythonSymbol(DomainModel):
    path: str
    module_name: str = Field(min_length=1, max_length=1_024)
    qualified_name: str = Field(min_length=1, max_length=2_048)
    name: str = Field(min_length=1, max_length=512)
    kind: PythonSymbolKind
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    evidence_id: EvidenceId

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_path(value)

    @model_validator(mode="after")
    def validate_lines(self) -> "PythonSymbol":
        if self.line_end < self.line_start:
            raise ValueError("line_end 不能小于 line_start")
        return self


class PythonImport(DomainModel):
    path: str
    importer_module: str = Field(min_length=1, max_length=1_024)
    imported_module: str = Field(min_length=1, max_length=1_024)
    imported_name: str | None = Field(default=None, min_length=1, max_length=512)
    level: int = Field(default=0, ge=0)
    line: int = Field(ge=1)
    evidence_id: EvidenceId

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_path(value)


class ModuleDependency(DomainModel):
    importer_module: str = Field(min_length=1, max_length=1_024)
    imported_module: str = Field(min_length=1, max_length=1_024)
    path: str
    line: int = Field(ge=1)
    evidence_id: EvidenceId

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_path(value)


class RepositoryParseIssue(DomainModel):
    path: str
    message: str = Field(min_length=1, max_length=2_000)
    line: int | None = Field(default=None, ge=1)
    evidence_id: EvidenceId

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_path(value)


class ValidationCommand(DomainModel):
    command_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9_.-]+$")
    argv: tuple[str, ...] = Field(min_length=1)
    source: ValidationCommandSource

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or "\x00" in value for value in values):
            raise ValueError("argv 不能包含空参数或 NUL")
        return values


class BaselineCheckResult(DomainModel):
    command: ValidationCommand
    exit_code: int
    stdout: str = Field(default="", max_length=20_000)
    stderr: str = Field(default="", max_length=20_000)
    timed_out: bool = False
    evidence_id: EvidenceId

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class RepositoryEvidence(DomainModel):
    evidence_id: EvidenceId
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    summary: str = Field(min_length=1, max_length=2_000)
    path: str | None = None
    symbol: str | None = Field(default=None, min_length=1, max_length=2_048)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_optional_path(cls, value: str | None) -> str | None:
        return validate_repository_path(value) if value is not None else None

    @model_validator(mode="after")
    def validate_locator(self) -> "RepositoryEvidence":
        if self.line_end is not None and self.line_start is None:
            raise ValueError("line_end 存在时必须提供 line_start")
        if self.line_start is not None and self.path is None:
            raise ValueError("行号证据必须提供 path")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end 不能小于 line_start")
        return self


class RepositoryProfile(DomainModel):
    schema_version: int = Field(default=1, ge=1)
    analyzer_version: str = Field(min_length=1, max_length=64)
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    files: tuple[RepositoryFile, ...]
    symbols: tuple[PythonSymbol, ...]
    imports: tuple[PythonImport, ...]
    dependencies: tuple[ModuleDependency, ...]
    validation_commands: tuple[ValidationCommand, ...]
    baseline_results: tuple[BaselineCheckResult, ...] = ()
    parse_issues: tuple[RepositoryParseIssue, ...] = ()
    evidence: tuple[RepositoryEvidence, ...] = ()
    profile_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "RepositoryProfile":
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("RepositoryProfile.files 路径不能重复")
        command_ids = [item.command_id for item in self.validation_commands]
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("validation_commands.command_id 不能重复")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id 不能重复")
        return self


class RequirementClues(DomainModel):
    terms: tuple[str, ...]
    symbols: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()

    @field_validator("terms", "symbols", "paths")
    @classmethod
    def validate_unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return validate_unique_strings(values, "检索线索")


class SearchHit(DomainModel):
    path: str
    line: int = Field(ge=1)
    term: str = Field(min_length=1, max_length=512)
    excerpt: str = Field(min_length=1, max_length=2_000)
    evidence_id: EvidenceId

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_path(value)


class ImpactCandidate(DomainModel):
    path: str
    score: int = Field(ge=1)
    reasons: tuple[str, ...] = Field(min_length=1)
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_repository_path(value)


class ImpactAnalysis(DomainModel):
    base_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    clues: RequirementClues
    search_hits: tuple[SearchHit, ...]
    candidates: tuple[ImpactCandidate, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguity: float = Field(ge=0.0, le=1.0)
    dependency_fan_out: int = Field(ge=0)
    risk_signals: tuple[str, ...] = ()


class RepositoryTaskAssessment(DomainModel):
    difficulty: DifficultyAssessment
    impact: ImpactAnalysis
    model_requirement: ModelRequirement

    @model_validator(mode="after")
    def validate_difficulty(self) -> "RepositoryTaskAssessment":
        if self.model_requirement.difficulty is not self.difficulty.difficulty:
            raise ValueError("ModelRequirement 难度必须与仓库难度评估一致")
        return self


def difficulty_rank(value: TaskDifficulty) -> int:
    return {
        TaskDifficulty.LOW: 0,
        TaskDifficulty.MEDIUM: 1,
        TaskDifficulty.HIGH: 2,
    }[value]
