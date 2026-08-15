"""Stable domain error codes and exceptions."""

from collections.abc import Mapping
from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_SPEC = "invalid_spec"
    INVALID_STATE_TRANSITION = "invalid_state_transition"
    INVALID_GRAPH = "invalid_graph"
    POLICY_REJECTED = "policy_rejected"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_CAPABILITY_MISMATCH = "model_capability_mismatch"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    CAPABILITY_NOT_FOUND = "capability_not_found"
    CAPABILITY_DENIED = "capability_denied"
    COMMAND_DENIED = "command_denied"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    RUNTIME_LIMIT_EXCEEDED = "runtime_limit_exceeded"
    NOT_GIT_REPOSITORY = "not_git_repository"
    GIT_COMMAND_FAILED = "git_command_failed"
    REPOSITORY_OBJECT_NOT_FOUND = "repository_object_not_found"
    REPOSITORY_LIMIT_EXCEEDED = "repository_limit_exceeded"
    BASELINE_EXECUTION_FAILED = "baseline_execution_failed"
    PROFILE_INTEGRITY_ERROR = "profile_integrity_error"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_INTEGRITY_ERROR = "artifact_integrity_error"
    WORKSPACE_ACCESS_DENIED = "workspace_access_denied"
    WORKSPACE_LEASE_NOT_FOUND = "workspace_lease_not_found"
    WORKSPACE_STATE_INVALID = "workspace_state_invalid"
    WORKTREE_OPERATION_FAILED = "worktree_operation_failed"
    PATCH_REJECTED = "patch_rejected"
    PATCH_EMPTY = "patch_empty"
    PATCH_BASE_MISMATCH = "patch_base_mismatch"
    PATCH_CONFLICT = "patch_conflict"
    INTEGRATION_STATE_INVALID = "integration_state_invalid"
    VALIDATION_FAILED = "validation_failed"
    COMMAND_TIMEOUT = "command_timeout"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    APPROVAL_REQUIRED = "approval_required"
    SCRIPT_EXHAUSTED = "script_exhausted"


class DomainError(Exception):
    """Base exception with a stable machine-readable code."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})
