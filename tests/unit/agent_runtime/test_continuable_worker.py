"""Deterministic tests for the continuable worker mechanism (dsh-inspired)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from evoweave_ds.agent_runtime.context_builder import ContextBuilder
from evoweave_ds.agent_runtime.runtime import WorkerRuntime
from evoweave_ds.capabilities.builtins import default_capabilities
from evoweave_ds.capabilities.registry import CapabilityRegistry
from evoweave_ds.capabilities.tool_executor import ToolExecutor
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import InputModality, ModelAvailability, ModelTier, ResultStatus
from evoweave_ds.domain.errors import ErrorCode
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelProfile, ModelRoutingDecision
from evoweave_ds.domain.ports import ModelResponse
from evoweave_ds.infrastructure.artifacts.memory import InMemoryArtifactStore
from evoweave_ds.infrastructure.models.fake import ScriptedModelGateway
from evoweave_ds.infrastructure.workspaces.fake import FakeWorkspace, FakeWorkspaceProvider


def _profile() -> ModelProfile:
    return ModelProfile(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        tier=ModelTier.LOW,
        availability=ModelAvailability.AVAILABLE,
        input_modalities=(InputModality.TEXT,),
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _routing() -> ModelRoutingDecision:
    return ModelRoutingDecision(
        decision_id=SpecId.new(),
        requirement_id=SpecId.new(),
        requirement_version=1,
        selected_model_key="deepseek:deepseek-v4-flash",
        selected_availability=ModelAvailability.AVAILABLE,
        reason="测试",
    )


_TASK_ID = TaskId.new()


def _spec(version: int = 1, continuable: bool = False) -> AgentExecutionSpec:
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=_TASK_ID,
        task_spec_id=SpecId.new(),
        task_spec_version=1,
        base_commit="c40a5a78bc3e0b549932129713ac65806815c579",
        goal="完成任务",
        acceptance_criteria=("完成",),
        required_modalities=(InputModality.TEXT,),
        model_routing=_routing(),
        tool_names=("file.read",),
        read_scope=("src",),
        write_scope=("src",),
        version=version,
        continuable=continuable,
    )


def _runtime(gateway: ScriptedModelGateway) -> WorkerRuntime:
    store = InMemoryArtifactStore()
    workspace = FakeWorkspace(
        task_id=_TASK_ID,
        files={"src/app.py": "content"},
        read_scope=("src",),
        write_scope=("src",),
    )
    return WorkerRuntime(
        model_gateway=gateway,
        tool_executor=ToolExecutor(CapabilityRegistry(default_capabilities())),
        context_builder=ContextBuilder(store),
        artifact_store=store,
        workspace_provider=FakeWorkspaceProvider({_TASK_ID: workspace}),
        event_recorder=_NoopRecorder(),
    )


class _NoopRecorder:
    def record(self, run_id, task_id, event_type, payload) -> None:
        pass


class TestContinuableWorker:
    def test_resume_requires_continuable(self) -> None:
        gateway = ScriptedModelGateway(profiles=(_profile(),))
        runtime = _runtime(gateway)
        failed = _failed_result()
        result = runtime.execute(_spec(), resume_from=failed)
        # 非 continuable 规格携带 resume_from 时结构化失败(不抛异常)。
        assert result.status is ResultStatus.FAILED
        assert result.failure is not None
        assert result.failure.code is ErrorCode.INVALID_SPEC

    def test_resume_injects_diagnostics(self) -> None:
        # 第一次失败(finish 失败), 第二次先读文件产生证据再成功;
        # 续接上下文应携带失败诊断。
        finish_fail = _finish_response("失败", failure_code=ErrorCode.MODEL_UNAVAILABLE)
        read_call = ModelResponse(
            model_key="deepseek:deepseek-v4-flash",
            text=json.dumps(
                {
                    "action": "tool",
                    "tool_name": "file.read",
                    "arguments": {"path": "src/app.py"},
                },
                ensure_ascii=False,
            ),
            input_tokens=10,
            output_tokens=5,
        )
        finish_ok = _finish_response("完成")
        gateway = ScriptedModelGateway(
            profiles=(_profile(),),
            responses=(finish_fail, read_call, finish_ok),
        )
        runtime = _runtime(gateway)
        first = runtime.execute(_spec(version=1))
        assert first.status is ResultStatus.FAILED
        second = runtime.execute(_spec(version=2, continuable=True), resume_from=first)
        assert second.status is ResultStatus.SUCCEEDED
        # 第二次请求的消息里必须包含续接上下文(失败诊断)。
        second_request = gateway.requests[-1]
        joined = "\n".join(second_request.messages)
        assert "续接上下文" in joined
        assert "model_unavailable" in joined


def _failed_result() -> object:
    from evoweave_ds.domain.resources import ResourceUsage
    from evoweave_ds.domain.task_result import TaskFailure, TaskResult

    return TaskResult(
        task_id=TaskId.new(),
        agent_id=AgentId.new(),
        execution_spec_id=SpecId.new(),
        execution_spec_version=1,
        status=ResultStatus.FAILED,
        summary="失败",
        failure=TaskFailure(code=ErrorCode.MODEL_UNAVAILABLE, message="失败"),
        usage=ResourceUsage(),
    )


def _finish_response(summary: str, failure_code: ErrorCode | None = None) -> ModelResponse:
    import json

    decision: dict[str, object] = {
        "action": "finish",
        "status": "succeeded" if failure_code is None else "failed",
        "summary": summary,
    }
    if failure_code is not None:
        decision["failure_code"] = failure_code.value
        decision["failure_message"] = "测试失败"
    return ModelResponse(
        model_key="deepseek:deepseek-v4-flash",
        text=json.dumps(decision, ensure_ascii=False),
        input_tokens=10,
        output_tokens=5,
    )
