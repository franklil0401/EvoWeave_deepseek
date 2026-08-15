"""Behavioral tests for one universal WorkerRuntime under different specs."""

import json
from collections.abc import Callable

from evoweave_ds.agent_runtime.context_builder import ContextBuilder
from evoweave_ds.agent_runtime.runtime import WorkerRuntime
from evoweave_ds.capabilities.builtins import default_capabilities
from evoweave_ds.capabilities.registry import CapabilityRegistry
from evoweave_ds.capabilities.tool_executor import ToolExecutor
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import ArtifactKind, EventType, InputModality, ResultStatus
from evoweave_ds.domain.errors import ErrorCode
from evoweave_ds.domain.identifiers import AgentId, ArtifactId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelRoutingDecision
from evoweave_ds.domain.ports import CommandResult, ModelResponse, WorkerAdapter
from evoweave_ds.domain.resources import RuntimeLimits
from evoweave_ds.infrastructure.artifacts.memory import InMemoryArtifactStore
from evoweave_ds.infrastructure.commands.fake import ScriptedCommandRunner
from evoweave_ds.infrastructure.models.fake import ScriptedModelGateway
from evoweave_ds.infrastructure.telemetry.memory import InMemoryEventRecorder
from evoweave_ds.infrastructure.workspaces.fake import FakeWorkspace, FakeWorkspaceProvider

MODEL_KEY = "fake:worker"


def _tool(name: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"action": "tool", "tool_name": name, "arguments": arguments},
        ensure_ascii=False,
    )


def _finish(summary: str = "任务完成") -> str:
    return json.dumps(
        {"action": "finish", "status": "succeeded", "summary": summary},
        ensure_ascii=False,
    )


def _failed_finish() -> str:
    return json.dumps(
        {
            "action": "finish",
            "status": "failed",
            "summary": "任务无法完成",
            "failure_code": "invalid_spec",
            "failure_message": "测试结束",
        },
        ensure_ascii=False,
    )


def _response(
    text: str,
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> ModelResponse:
    return ModelResponse(
        model_key=MODEL_KEY,
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _spec(
    *,
    run_id: RunId,
    task_id: TaskId,
    tools: tuple[str, ...],
    write_scope: tuple[str, ...] = (),
    commands: tuple[str, ...] = (),
    limits: RuntimeLimits | None = None,
    modalities: tuple[InputModality, ...] = (InputModality.TEXT,),
    input_ids: tuple[ArtifactId, ...] = (),
    reasoning_effort: str = "low",
) -> AgentExecutionSpec:
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=run_id,
        agent_id=AgentId.new(),
        task_id=task_id,
        task_spec_id=SpecId.new(),
        task_spec_version=1,
        base_commit="a" * 40,
        goal="执行动态规格",
        acceptance_criteria=("产出结构化结果",),
        required_modalities=modalities,
        model_routing=ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=SpecId.new(),
            requirement_version=1,
            selected_model_key=MODEL_KEY,
            reasoning_effort=reasoning_effort,
            reason="测试路由",
        ),
        tool_names=tools,
        allowed_commands=commands,
        read_scope=("src",),
        write_scope=write_scope,
        input_artifact_ids=input_ids,
        runtime_limits=limits or RuntimeLimits(max_steps=8, max_tool_calls=4),
    )


def _runtime(
    *,
    responses: tuple[ModelResponse, ...],
    workspaces: dict[TaskId, FakeWorkspace],
    store: InMemoryArtifactStore,
    recorder: InMemoryEventRecorder,
    runner: ScriptedCommandRunner | None = None,
    clock: Callable[[], float] | None = None,
    gateway: ScriptedModelGateway | None = None,
) -> WorkerRuntime:
    selected_gateway = gateway or ScriptedModelGateway(responses=responses)
    tool_executor = ToolExecutor(CapabilityRegistry(default_capabilities()))
    context_builder = ContextBuilder(store)
    workspace_provider = FakeWorkspaceProvider(workspaces)
    if clock is None:
        return WorkerRuntime(
            model_gateway=selected_gateway,
            tool_executor=tool_executor,
            context_builder=context_builder,
            artifact_store=store,
            workspace_provider=workspace_provider,
            event_recorder=recorder,
            command_runner=runner,
        )
    return WorkerRuntime(
        model_gateway=selected_gateway,
        tool_executor=tool_executor,
        context_builder=context_builder,
        artifact_store=store,
        workspace_provider=workspace_provider,
        event_recorder=recorder,
        command_runner=runner,
        clock=clock,
    )


def test_one_runtime_executes_locator_modifier_and_validator_specs() -> None:
    run_id = RunId.new()
    locate_id, modify_id, validate_id = TaskId.new(), TaskId.new(), TaskId.new()
    workspaces = {
        locate_id: FakeWorkspace(
            task_id=locate_id,
            files={"src/app.py": "target = 1\n"},
            read_scope=("src",),
        ),
        modify_id: FakeWorkspace(
            task_id=modify_id,
            files={"src/app.py": "value = 1\n"},
            read_scope=("src",),
            write_scope=("src",),
        ),
        validate_id: FakeWorkspace(
            task_id=validate_id,
            files={"src/app.py": "value = 2\n"},
            read_scope=("src",),
        ),
    }
    command = ("pytest", "-q")
    runner = ScriptedCommandRunner(
        {command: CommandResult(argv=command, exit_code=0, stdout="3 passed")}
    )
    store = InMemoryArtifactStore()
    recorder = InMemoryEventRecorder()
    responses = (
        _response(_tool("file.search", {"query": "target", "prefix": "src"})),
        _response(_finish("MODEL_RAW_SENTINEL")),
        _response(_tool("file.write", {"path": "src/app.py", "content": "value = 2\n"})),
        _response(_finish("已生成补丁")),
        _response(_tool("command.run", {"argv": list(command)})),
        _response(_finish("验证通过")),
    )
    gateway = ScriptedModelGateway(responses=responses)
    runtime = _runtime(
        responses=(),
        workspaces=workspaces,
        store=store,
        recorder=recorder,
        runner=runner,
        gateway=gateway,
    )
    assert isinstance(runtime, WorkerAdapter)

    locate = runtime.execute(
        _spec(run_id=run_id, task_id=locate_id, tools=("file.search", "file.read"))
    )
    modify = runtime.execute(
        _spec(
            run_id=run_id,
            task_id=modify_id,
            tools=("file.read", "file.write"),
            write_scope=("src",),
        )
    )
    validate = runtime.execute(
        _spec(
            run_id=run_id,
            task_id=validate_id,
            tools=("command.run",),
            commands=("pytest",),
        )
    )

    assert [item.status for item in (locate, modify, validate)] == [
        ResultStatus.SUCCEEDED,
        ResultStatus.SUCCEEDED,
        ResultStatus.SUCCEEDED,
    ]
    assert workspaces[modify_id].read_text("src/app.py") == "value = 2\n"
    assert modify.artifacts[0].kind is ArtifactKind.PATCH
    assert validate.artifacts[0].kind is ArtifactKind.COMMAND_LOG
    events = recorder.events_for(run_id)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert sum(event.event_type is EventType.AGENT_STARTED for event in events) == 3
    assert sum(event.event_type is EventType.ARTIFACT_PERSISTED for event in events) == 2
    assert "MODEL_RAW_SENTINEL" not in "".join(event.model_dump_json() for event in events)
    locator_prompt = "\n".join(gateway.requests[0].messages)
    modifier_prompt = "\n".join(gateway.requests[2].messages)
    validator_prompt = "\n".join(gateway.requests[4].messages)
    assert "file.search" in locator_prompt
    assert "file.write" not in locator_prompt
    assert "file.write" in modifier_prompt
    assert "command.run" not in modifier_prompt
    assert "command.run" in validator_prompt
    assert "file.write" not in validator_prompt
    assert "Worker 决策 JSON Schema" in locator_prompt


def test_ungranted_tool_is_rejected_and_model_can_self_correct() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    workspace = FakeWorkspace(
        task_id=task_id,
        files={"src/app.py": "old\n"},
        read_scope=("src",),
        write_scope=("src",),
    )
    recorder = InMemoryEventRecorder()
    runtime = _runtime(
        responses=(
            _response(_tool("file.write", {"path": "src/app.py", "content": "bad"})),
            _response(_tool("file.read", {"path": "src/app.py"})),
            _response(_finish()),
        ),
        workspaces={task_id: workspace},
        store=InMemoryArtifactStore(),
        recorder=recorder,
    )
    result = runtime.execute(_spec(run_id=run_id, task_id=task_id, tools=("file.read",)))
    assert result.status is ResultStatus.SUCCEEDED
    assert result.failure is None
    assert workspace.read_text("src/app.py") == "old\n"
    assert EventType.TOOL_REJECTED in {event.event_type for event in recorder.events_for(run_id)}


def test_escaping_search_path_is_rejected_and_model_can_retry() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    gateway = ScriptedModelGateway(
        responses=(
            _response(_tool("file.search", {"query": "value", "prefix": "../"})),
            _response(_tool("file.search", {"query": "value", "prefix": "src"})),
            _response(_finish()),
        )
    )
    recorder = InMemoryEventRecorder()
    runtime = _runtime(
        responses=(),
        workspaces={
            task_id: FakeWorkspace(
                task_id=task_id,
                files={"src/app.py": "value = 1\n"},
                read_scope=("src",),
            )
        },
        store=InMemoryArtifactStore(),
        recorder=recorder,
        gateway=gateway,
    )

    result = runtime.execute(_spec(run_id=run_id, task_id=task_id, tools=("file.search",)))

    assert result.status is ResultStatus.SUCCEEDED
    assert any("工具拒绝观察" in message for message in gateway.requests[1].messages)
    assert EventType.TOOL_REJECTED in {event.event_type for event in recorder.events_for(run_id)}


def test_invalid_model_output_becomes_structured_failure() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    recorder = InMemoryEventRecorder()
    runtime = _runtime(
        responses=(_response("not-json"), _response("still-not-json")),
        workspaces={task_id: FakeWorkspace(task_id=task_id, read_scope=("src",))},
        store=InMemoryArtifactStore(),
        recorder=recorder,
    )
    result = runtime.execute(_spec(run_id=run_id, task_id=task_id, tools=("file.read",)))
    assert result.failure is not None
    assert result.failure.code is ErrorCode.INVALID_MODEL_OUTPUT
    rejected = [
        event
        for event in recorder.events_for(run_id)
        if event.event_type is EventType.MODEL_OUTPUT_REJECTED
    ]
    assert len(rejected) == 1


def test_invalid_model_output_gets_one_bounded_self_correction() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    recorder = InMemoryEventRecorder()
    gateway = ScriptedModelGateway(
        responses=(
            _response("先分析一下，稍后再给决定"),
            _response(_tool("file.read", {"path": "src/app.py"})),
            _response(_finish()),
        )
    )
    runtime = _runtime(
        responses=(),
        workspaces={
            task_id: FakeWorkspace(
                task_id=task_id,
                files={"src/app.py": "value = 1\n"},
                read_scope=("src",),
            )
        },
        store=InMemoryArtifactStore(),
        recorder=recorder,
        gateway=gateway,
    )

    result = runtime.execute(_spec(run_id=run_id, task_id=task_id, tools=("file.read",)))

    assert result.status is ResultStatus.SUCCEEDED
    assert "协议拒绝观察" in gateway.requests[1].messages[-1]
    assert (
        sum(
            event.event_type is EventType.MODEL_OUTPUT_REJECTED
            for event in recorder.events_for(run_id)
        )
        == 1
    )


def test_success_without_evidence_is_rejected() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    runtime = _runtime(
        responses=(_response(_finish()),),
        workspaces={task_id: FakeWorkspace(task_id=task_id, read_scope=("src",))},
        store=InMemoryArtifactStore(),
        recorder=InMemoryEventRecorder(),
    )
    result = runtime.execute(_spec(run_id=run_id, task_id=task_id, tools=("file.read",)))
    assert result.failure is not None
    assert result.failure.code is ErrorCode.INVALID_MODEL_OUTPUT


def test_step_limit_stops_loop_before_second_model_call() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    gateway_responses = (
        _response(_tool("file.read", {"path": "src/app.py"})),
        _response(_failed_finish()),
    )
    runtime = _runtime(
        responses=gateway_responses,
        workspaces={
            task_id: FakeWorkspace(
                task_id=task_id,
                files={"src/app.py": "value = 1\n"},
                read_scope=("src",),
            )
        },
        store=InMemoryArtifactStore(),
        recorder=InMemoryEventRecorder(),
    )
    result = runtime.execute(
        _spec(
            run_id=run_id,
            task_id=task_id,
            tools=("file.read",),
            limits=RuntimeLimits(max_steps=1, max_tool_calls=2),
        )
    )
    assert result.failure is not None
    assert result.failure.code is ErrorCode.RUNTIME_LIMIT_EXCEEDED
    assert result.usage.steps == 2


def test_token_limit_stops_before_parsing_model_decision() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    runtime = _runtime(
        responses=(_response(_failed_finish(), output_tokens=11),),
        workspaces={task_id: FakeWorkspace(task_id=task_id, read_scope=("src",))},
        store=InMemoryArtifactStore(),
        recorder=InMemoryEventRecorder(),
    )
    result = runtime.execute(
        _spec(
            run_id=run_id,
            task_id=task_id,
            tools=("file.read",),
            limits=RuntimeLimits(max_output_tokens=10),
        )
    )
    assert result.failure is not None
    assert result.failure.code is ErrorCode.RUNTIME_LIMIT_EXCEEDED


def test_input_token_limit_uses_gateway_usage() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    runtime = _runtime(
        responses=(_response(_failed_finish(), input_tokens=10_001),),
        workspaces={task_id: FakeWorkspace(task_id=task_id, read_scope=("src",))},
        store=InMemoryArtifactStore(),
        recorder=InMemoryEventRecorder(),
    )
    result = runtime.execute(
        _spec(
            run_id=run_id,
            task_id=task_id,
            tools=("file.read",),
            limits=RuntimeLimits(max_input_tokens=10_000),
        )
    )
    assert result.failure is not None
    assert result.failure.code is ErrorCode.RUNTIME_LIMIT_EXCEEDED


def test_context_estimate_stops_before_model_call() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    gateway = ScriptedModelGateway(responses=(_response(_failed_finish()),))
    runtime = _runtime(
        responses=(),
        workspaces={task_id: FakeWorkspace(task_id=task_id, read_scope=("src",))},
        store=InMemoryArtifactStore(),
        recorder=InMemoryEventRecorder(),
        gateway=gateway,
    )
    result = runtime.execute(
        _spec(
            run_id=run_id,
            task_id=task_id,
            tools=("file.read",),
            limits=RuntimeLimits(max_input_tokens=1),
        )
    )
    assert result.failure is not None
    assert result.failure.code is ErrorCode.CONTEXT_LIMIT_EXCEEDED
    assert not gateway.requests


def test_tool_call_limit_stops_before_capability_execution() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    workspace = FakeWorkspace(
        task_id=task_id,
        files={"src/app.py": "old\n"},
        read_scope=("src",),
        write_scope=("src",),
    )
    runtime = _runtime(
        responses=(_response(_tool("file.write", {"path": "src/app.py", "content": "new"})),),
        workspaces={task_id: workspace},
        store=InMemoryArtifactStore(),
        recorder=InMemoryEventRecorder(),
    )
    result = runtime.execute(
        _spec(
            run_id=run_id,
            task_id=task_id,
            tools=("file.write",),
            write_scope=("src",),
            limits=RuntimeLimits(max_tool_calls=0),
        )
    )
    assert result.failure is not None
    assert result.failure.code is ErrorCode.RUNTIME_LIMIT_EXCEEDED
    assert workspace.read_text("src/app.py") == "old\n"


def test_timeout_is_deterministic_with_injected_clock() -> None:
    run_id, task_id = RunId.new(), TaskId.new()
    ticks = iter((0.0, 2.0, 2.0, 2.0))
    runtime = _runtime(
        responses=(_response(_failed_finish()),),
        workspaces={task_id: FakeWorkspace(task_id=task_id, read_scope=("src",))},
        store=InMemoryArtifactStore(),
        recorder=InMemoryEventRecorder(),
        clock=lambda: next(ticks),
    )
    result = runtime.execute(
        _spec(
            run_id=run_id,
            task_id=task_id,
            tools=("file.read",),
            limits=RuntimeLimits(timeout_seconds=1),
        )
    )
    assert result.failure is not None
    assert result.failure.code is ErrorCode.RUNTIME_LIMIT_EXCEEDED


def test_runtime_passes_reasoning_effort_to_gateway() -> None:
    store = InMemoryArtifactStore()
    run_id = RunId.new()
    task_id = TaskId.new()
    gateway = ScriptedModelGateway(
        responses=(_response(_failed_finish()), _response(_failed_finish()))
    )
    runtime = _runtime(
        responses=(),
        workspaces={task_id: FakeWorkspace(task_id=task_id, read_scope=("src",))},
        store=store,
        recorder=InMemoryEventRecorder(),
        gateway=gateway,
    )
    runtime.execute(
        _spec(
            run_id=run_id,
            task_id=task_id,
            tools=("file.read",),
            reasoning_effort="high",
        )
    )
    assert gateway.requests[0].reasoning_effort == "high"
