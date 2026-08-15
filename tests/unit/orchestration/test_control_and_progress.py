import pytest

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.artifacts import EvidenceRef
from evoweave_ds.domain.enums import (
    EvidenceKind,
    InputModality,
    ModelAvailability,
    ResultStatus,
    TaskDifficulty,
)
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, EvidenceId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import (
    DifficultyAssessment,
    ModelRequirement,
    ModelRoutingDecision,
)
from evoweave_ds.domain.task_result import TaskResult
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.orchestration.control_view import TaskSuggestion
from evoweave_ds.orchestration.progress_detector import ProgressDetector
from evoweave_ds.orchestration.result_reducer import ResultReducer


def test_result_reducer_keeps_metadata_but_not_raw_evidence_or_logs() -> None:
    execution = _execution_spec()
    result = TaskResult(
        task_id=execution.task_id,
        agent_id=execution.agent_id,
        execution_spec_id=execution.spec_id,
        execution_spec_version=execution.version,
        status=ResultStatus.SUCCEEDED,
        summary="完成" * 2_000,
        evidence=(
            EvidenceRef(
                evidence_id=EvidenceId.new(),
                kind=EvidenceKind.FILE,
                summary="源码证据",
                repository_path="src/app.py",
            ),
        ),
    )
    summary = ResultReducer().reduce(result=result, execution_spec=execution)

    serialized = summary.model_dump_json()
    assert len(summary.summary) == 2_000
    assert "src/app.py" not in serialized
    assert "源码证据" not in serialized
    assert not hasattr(summary, "evidence")


def test_worker_suggestion_cannot_create_task_and_must_bind_source() -> None:
    execution = _execution_spec()
    result = TaskResult(
        task_id=execution.task_id,
        agent_id=execution.agent_id,
        execution_spec_id=execution.spec_id,
        execution_spec_version=execution.version,
        status=ResultStatus.SUCCEEDED,
        summary="建议拆分",
        evidence=(
            EvidenceRef(
                evidence_id=EvidenceId.new(),
                kind=EvidenceKind.FILE,
                summary="证据",
                repository_path="src/app.py",
            ),
        ),
    )
    suggestion = TaskSuggestion(
        suggestion_id=SpecId.new(),
        source_task_id=TaskId.new(),
        goal="新增任务",
        rationale="发现独立范围",
    )
    with pytest.raises(DomainError) as error:
        ResultReducer().reduce(
            result=result,
            execution_spec=execution,
            suggestions=(suggestion,),
        )
    assert error.value.code is ErrorCode.INVALID_SPEC


def test_progress_detector_rejects_semantically_duplicate_task_specs() -> None:
    task = _task_spec()
    duplicate = task.model_copy(
        update={
            "spec_id": SpecId.new(),
            "task_id": TaskId.new(),
            "model_requirement": task.model_requirement.model_copy(
                update={"requirement_id": SpecId.new(), "task_id": TaskId.new()}
            ),
        }
    )
    duplicate = duplicate.model_copy(
        update={
            "model_requirement": duplicate.model_requirement.model_copy(
                update={"task_id": duplicate.task_id}
            )
        }
    )
    with pytest.raises(DomainError) as error:
        ProgressDetector().reject_duplicate_specs((duplicate,), (task,))
    assert error.value.code is ErrorCode.POLICY_REJECTED


def _execution_spec() -> AgentExecutionSpec:
    task_id = TaskId.new()
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=task_id,
        task_spec_id=SpecId.new(),
        task_spec_version=1,
        base_commit="a" * 40,
        goal="执行任务",
        acceptance_criteria=("完成",),
        model_routing=ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=SpecId.new(),
            requirement_version=1,
            selected_model_key="fake:text",
            selected_availability=ModelAvailability.AVAILABLE,
            reason="测试",
        ),
        read_scope=("src",),
    )


def _task_spec() -> TaskSpec:
    task_id = TaskId.new()
    difficulty = DifficultyAssessment(difficulty=TaskDifficulty.LOW, rationale="简单")
    requirement = ModelRequirement(
        requirement_id=SpecId.new(),
        task_id=task_id,
        difficulty=TaskDifficulty.LOW,
        required_modalities=(InputModality.TEXT,),
    )
    return TaskSpec(
        spec_id=SpecId.new(),
        task_id=task_id,
        change_spec_id=SpecId.new(),
        goal="修改同一个目标",
        base_commit="a" * 40,
        acceptance_criteria=("完成",),
        read_scope=("src",),
        write_scope=("src/app.py",),
        difficulty=difficulty,
        model_requirement=requirement,
    )
