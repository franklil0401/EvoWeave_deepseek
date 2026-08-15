"""Closed vocabularies for EvoWeave domain contracts."""

from enum import StrEnum


class ArtifactKind(StrEnum):
    REPOSITORY_PROFILE = "repository_profile"
    CONTEXT_BUNDLE = "context_bundle"
    PATCH = "patch"
    TEST_REPORT = "test_report"
    COMMAND_LOG = "command_log"
    CONTROL_SUMMARY = "control_summary"
    GENERIC = "generic"


class ArtifactSecurityStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ArtifactSource(StrEnum):
    LOCAL_FILE = "local_file"
    CONTROLLED_INGESTION = "controlled_ingestion"
    GENERATED = "generated"


class EvidenceKind(StrEnum):
    FILE = "file"
    SYMBOL = "symbol"
    COMMAND = "command"
    ARTIFACT = "artifact"


class InputModality(StrEnum):
    TEXT = "text"


class ModelAvailability(StrEnum):
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ModelTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CapabilityAccess(StrEnum):
    READ = "read"
    WRITE = "write"
    COMMAND = "command"


class WorkspaceAccessMode(StrEnum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class WorkspaceLeaseStatus(StrEnum):
    CREATING = "creating"
    ACTIVE = "active"
    RELEASING = "releasing"
    RELEASED = "released"
    FAILED = "failed"
    ORPHANED = "orphaned"


class TaskLeaseStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    RELEASED = "released"
    EXPIRED = "expired"


class IntegrationStatus(StrEnum):
    CREATING = "creating"
    ACTIVE = "active"
    RELEASING = "releasing"
    RELEASED = "released"
    FAILED = "failed"


class RunStatus(StrEnum):
    INITIALIZED = "initialized"
    ANALYZED = "analyzed"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"


class PatchConflictKind(StrEnum):
    BASE_MISMATCH = "base_mismatch"
    INTEGRITY = "integrity"
    SYNTAX = "syntax"
    PATH_SCOPE = "path_scope"
    SENSITIVE_PATH = "sensitive_path"
    WRITE_SET = "write_set"
    APPLY = "apply"


class ValidationPhase(StrEnum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"
    CANDIDATE_RETRY = "candidate_retry"


class ValidationScope(StrEnum):
    LOCAL = "local"
    IMPACT = "impact"
    FULL = "full"
    LINT = "lint"


class FailureClassification(StrEnum):
    PRE_EXISTING = "pre_existing"
    NEW = "new"
    RESOLVED = "resolved"
    UNSTABLE = "unstable"


class TaskDifficulty(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskRelation(StrEnum):
    DEPENDS_ON = "depends_on"
    VALIDATES = "validates"
    SUPERSEDES = "supersedes"


class PolicyViolationCode(StrEnum):
    TOO_MANY_NODES = "too_many_nodes"
    TOO_MANY_RUNNING_TASKS = "too_many_running_tasks"
    GRAPH_INVALID = "graph_invalid"
    RETRY_LIMIT_EXCEEDED = "retry_limit_exceeded"
    WRITE_SCOPE_OVERLAP = "write_scope_overlap"
    TOO_MANY_TASKS_PER_DECISION = "too_many_tasks_per_decision"
    NO_PROGRESS = "no_progress"
    DUPLICATE_TASK = "duplicate_task"


class EventType(StrEnum):
    RUN_CREATED = "run_created"
    TASK_GRAPH_REPLACED = "task_graph_replaced"
    TASK_STATUS_CHANGED = "task_status_changed"
    MODEL_ROUTED = "model_routed"
    AGENT_STARTED = "agent_started"
    AGENT_FINISHED = "agent_finished"
    ARTIFACT_PERSISTED = "artifact_persisted"
    POLICY_REJECTED = "policy_rejected"
    MODEL_CALL_COMPLETED = "model_call_completed"
    MODEL_OUTPUT_REJECTED = "model_output_rejected"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_REJECTED = "tool_rejected"
    ORCHESTRATION_DECIDED = "orchestration_decided"
    TASK_LEASED = "task_leased"
    CHECKPOINT_SAVED = "checkpoint_saved"
    RUN_FINISHED = "run_finished"
