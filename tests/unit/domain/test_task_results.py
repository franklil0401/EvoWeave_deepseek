"""Tests for structured worker result invariants."""

import pytest
from pydantic import ValidationError

from evoweave_ds.domain.artifacts import EvidenceRef
from evoweave_ds.domain.enums import EvidenceKind, ResultStatus
from evoweave_ds.domain.errors import ErrorCode
from evoweave_ds.domain.identifiers import AgentId, EvidenceId, SpecId, TaskId
from evoweave_ds.domain.task_result import TaskFailure, TaskResult


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        evidence_id=EvidenceId.new(),
        kind=EvidenceKind.FILE,
        summary="领域模型已经通过单元测试",
        repository_path="src/evoweave_ds/domain/base.py",
        line_start=1,
    )


def test_success_result_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="至少一条证据"):
        TaskResult(
            task_id=TaskId.new(),
            agent_id=AgentId.new(),
            execution_spec_id=SpecId.new(),
            execution_spec_version=1,
            status=ResultStatus.SUCCEEDED,
            summary="完成",
        )


def test_failure_result_requires_structured_failure() -> None:
    with pytest.raises(ValidationError, match="必须包含 failure"):
        TaskResult(
            task_id=TaskId.new(),
            agent_id=AgentId.new(),
            execution_spec_id=SpecId.new(),
            execution_spec_version=1,
            status=ResultStatus.FAILED,
            summary="执行失败",
        )


def test_valid_success_result_round_trips() -> None:
    result = TaskResult(
        task_id=TaskId.new(),
        agent_id=AgentId.new(),
        execution_spec_id=SpecId.new(),
        execution_spec_version=1,
        status=ResultStatus.SUCCEEDED,
        summary="执行成功",
        evidence=(_evidence(),),
    )
    assert TaskResult.model_validate_json(result.model_dump_json()) == result


def test_valid_failure_exposes_stable_error_code() -> None:
    result = TaskResult(
        task_id=TaskId.new(),
        agent_id=AgentId.new(),
        execution_spec_id=SpecId.new(),
        execution_spec_version=1,
        status=ResultStatus.FAILED,
        summary="模型暂时不可用",
        failure=TaskFailure(
            code=ErrorCode.MODEL_UNAVAILABLE,
            message="无可用模型",
            retryable=True,
        ),
    )
    assert result.failure is not None
    assert result.failure.retryable
