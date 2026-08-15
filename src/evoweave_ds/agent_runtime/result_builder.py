"""Build final TaskResult from trusted runtime observations."""

import json
from collections.abc import Iterable
from typing import cast

from pydantic import JsonValue

from evoweave_ds.agent_runtime.decisions import FinishDecision
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.artifacts import ArtifactRef, EvidenceRef
from evoweave_ds.domain.enums import ResultStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.resources import ResourceUsage
from evoweave_ds.domain.task_result import TaskFailure, TaskResult


class ResultBuilder:
    def build(
        self,
        *,
        spec: AgentExecutionSpec,
        decision: FinishDecision,
        evidence: Iterable[EvidenceRef],
        artifacts: Iterable[ArtifactRef],
        usage: ResourceUsage,
    ) -> TaskResult:
        unique_evidence = _unique_evidence(evidence)
        unique_artifacts = _unique_artifacts(artifacts)
        failure = None
        if decision.status is not ResultStatus.SUCCEEDED:
            if decision.failure_code is None or decision.failure_message is None:
                raise DomainError(ErrorCode.INVALID_MODEL_OUTPUT, "失败决定缺少结构化错误")
            failure = TaskFailure(
                code=decision.failure_code,
                message=decision.failure_message,
                retryable=decision.retryable,
            )
        try:
            return TaskResult(
                task_id=spec.task_id,
                agent_id=spec.agent_id,
                execution_spec_id=spec.spec_id,
                execution_spec_version=spec.version,
                status=decision.status,
                summary=decision.summary,
                evidence=unique_evidence,
                artifacts=unique_artifacts,
                risk_level=decision.risk_level,
                risk_notes=decision.risk_notes,
                usage=usage,
                failure=failure,
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.INVALID_MODEL_OUTPUT,
                "模型完成决定无法构成有效 TaskResult",
            ) from exc


def build_failure_result(
    *,
    spec: AgentExecutionSpec,
    error: DomainError,
    usage: ResourceUsage,
) -> TaskResult:
    return TaskResult(
        task_id=spec.task_id,
        agent_id=spec.agent_id,
        execution_spec_id=spec.spec_id,
        execution_spec_version=spec.version,
        status=ResultStatus.FAILED,
        summary=error.message,
        usage=usage,
        failure=TaskFailure(
            code=error.code,
            message=error.message,
            retryable=error.code in {ErrorCode.MODEL_UNAVAILABLE},
            details=cast(
                dict[str, JsonValue],
                json.loads(json.dumps(error.details, ensure_ascii=False, default=str)),
            ),
        ),
    )


def _unique_evidence(values: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    by_id = {value.evidence_id: value for value in values}
    return tuple(by_id.values())


def _unique_artifacts(values: Iterable[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    by_id = {value.artifact_id: value for value in values}
    return tuple(by_id.values())
