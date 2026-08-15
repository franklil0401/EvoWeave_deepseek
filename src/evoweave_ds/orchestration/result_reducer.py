"""Validate worker identity and reduce L2 results into compact L3 summaries."""

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.task_result import TaskResult
from evoweave_ds.orchestration.control_view import (
    ArtifactMetadata,
    ResultControlSummary,
    TaskSuggestion,
)


class ResultReducer:
    def reduce(
        self,
        *,
        result: TaskResult,
        execution_spec: AgentExecutionSpec,
        suggestions: tuple[TaskSuggestion, ...] = (),
    ) -> ResultControlSummary:
        if (
            result.task_id != execution_spec.task_id
            or result.agent_id != execution_spec.agent_id
            or result.execution_spec_id != execution_spec.spec_id
            or result.execution_spec_version != execution_spec.version
        ):
            raise DomainError(ErrorCode.INVALID_SPEC, "Worker 结果与执行规格身份不匹配")
        if any(item.source_task_id != result.task_id for item in suggestions):
            raise DomainError(ErrorCode.INVALID_SPEC, "任务建议必须绑定产生它的 Worker 任务")
        artifacts = tuple(
            ArtifactMetadata(
                artifact_id=artifact.artifact_id,
                kind=artifact.kind,
                media_type=artifact.media_type,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
            )
            for artifact in result.artifacts
        )
        return ResultControlSummary(
            task_id=result.task_id,
            agent_id=result.agent_id,
            execution_spec_id=result.execution_spec_id,
            status=result.status,
            summary=result.summary[:2_000],
            artifacts=artifacts,
            suggestions=suggestions,
        )
