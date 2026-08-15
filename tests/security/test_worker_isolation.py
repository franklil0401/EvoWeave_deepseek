"""Security regressions proving untrusted input cannot expand worker authority."""

import json

from evoweave_ds.agent_runtime.context_builder import ContextBuilder
from evoweave_ds.agent_runtime.runtime import WorkerRuntime
from evoweave_ds.capabilities.builtins import default_capabilities
from evoweave_ds.capabilities.registry import CapabilityRegistry
from evoweave_ds.capabilities.tool_executor import ToolExecutor
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import InputModality, ResultStatus
from evoweave_ds.domain.identifiers import AgentId, ArtifactId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelRoutingDecision
from evoweave_ds.domain.ports import ModelResponse
from evoweave_ds.infrastructure.artifacts.memory import InMemoryArtifactStore
from evoweave_ds.infrastructure.models.fake import ScriptedModelGateway
from evoweave_ds.infrastructure.telemetry.memory import InMemoryEventRecorder
from evoweave_ds.infrastructure.workspaces.fake import FakeWorkspace, FakeWorkspaceProvider

MODEL_KEY = "fake:vision"


def _spec(
    task_id: TaskId,
    *,
    tools: tuple[str, ...],
    input_ids: tuple[ArtifactId, ...] = (),
    modalities: tuple[InputModality, ...] = (InputModality.TEXT,),
) -> AgentExecutionSpec:
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=task_id,
        task_spec_id=SpecId.new(),
        task_spec_version=1,
        base_commit="a" * 40,
        goal="处理不可信输入",
        acceptance_criteria=("不扩大权限",),
        required_modalities=modalities,
        model_routing=ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=SpecId.new(),
            requirement_version=1,
            selected_model_key=MODEL_KEY,
            reason="安全测试",
        ),
        tool_names=tools,
        read_scope=("src",),
        input_artifact_ids=input_ids,
    )


def _runtime(
    *,
    responses: tuple[str, ...],
    task_id: TaskId,
    workspace: FakeWorkspace,
    store: InMemoryArtifactStore,
) -> WorkerRuntime:
    return WorkerRuntime(
        model_gateway=ScriptedModelGateway(
            responses=tuple(ModelResponse(model_key=MODEL_KEY, text=item) for item in responses)
        ),
        tool_executor=ToolExecutor(CapabilityRegistry(default_capabilities())),
        context_builder=ContextBuilder(store),
        artifact_store=store,
        workspace_provider=FakeWorkspaceProvider({task_id: workspace}),
        event_recorder=InMemoryEventRecorder(),
    )


def test_path_traversal_from_model_is_normalized_to_structured_denial() -> None:
    task_id = TaskId.new()
    store = InMemoryArtifactStore()
    responses = (
        json.dumps(
            {
                "action": "tool",
                "tool_name": "file.read",
                "arguments": {"path": "src/../secret"},
            }
        ),
        json.dumps(
            {"action": "tool", "tool_name": "file.read", "arguments": {"path": "src/app.py"}}
        ),
        json.dumps({"action": "finish", "status": "succeeded", "summary": "已纠正路径"}),
    )
    result = _runtime(
        responses=responses,
        task_id=task_id,
        workspace=FakeWorkspace(
            task_id=task_id,
            files={"src/app.py": "safe\n"},
            read_scope=("src",),
        ),
        store=store,
    ).execute(_spec(task_id, tools=("file.read",)))
    assert result.status is ResultStatus.SUCCEEDED
    assert result.failure is None


def test_model_output_cannot_grant_write_capability() -> None:
    store = InMemoryArtifactStore()
    task_id = TaskId.new()
    responses = (
        json.dumps(
            {
                "action": "tool",
                "tool_name": "file.write",
                "arguments": {"path": "src/app.py", "content": "compromised"},
            }
        ),
        json.dumps(
            {"action": "tool", "tool_name": "file.read", "arguments": {"path": "src/app.py"}}
        ),
        json.dumps({"action": "finish", "status": "succeeded", "summary": "保持只读权限"}),
    )
    workspace = FakeWorkspace(
        task_id=task_id,
        files={"src/app.py": "safe\n"},
        read_scope=("src",),
        write_scope=("src",),
    )
    result = _runtime(
        responses=responses,
        task_id=task_id,
        workspace=workspace,
        store=store,
    ).execute(
        _spec(
            task_id,
            tools=("file.read",),
        )
    )
    assert result.status is ResultStatus.SUCCEEDED
    assert result.failure is None
    assert workspace.read_text("src/app.py") == "safe\n"
