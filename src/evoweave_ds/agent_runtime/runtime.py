"""Universal role-free model → tool → observation → result loop."""

import json
from collections.abc import Callable
from time import monotonic

from pydantic import JsonValue

from evoweave_ds.agent_runtime.budget_tracker import RuntimeLimitTracker
from evoweave_ds.agent_runtime.context_builder import ContextBuilder
from evoweave_ds.agent_runtime.decisions import (
    FinishDecision,
    ToolCallDecision,
    parse_worker_decision,
    worker_decision_json_schema,
)
from evoweave_ds.agent_runtime.result_builder import ResultBuilder, build_failure_result
from evoweave_ds.capabilities.tool_executor import ToolExecutor
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.artifacts import ArtifactRef, EvidenceRef
from evoweave_ds.domain.enums import EventType
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.ports import (
    ArtifactStore,
    CommandRunner,
    EventRecorder,
    ModelGateway,
    ModelRequest,
    ModelResponse,
    WorkspaceProvider,
)
from evoweave_ds.domain.task_result import TaskResult

_SYSTEM_MESSAGE = (
    "你是 EvoWeave 的通用临时执行实例。只能调用执行规格授予的能力。"
    "每次只输出一个符合 JSON 协议的 tool 或 finish 决定；不要输出私有推理过程。"
    "授权修改已经满足目标后应立即 finish；不得反复写入相同内容。"
)

_RECOVERABLE_TOOL_ERRORS = {
    ErrorCode.INVALID_SPEC,
    ErrorCode.CAPABILITY_NOT_FOUND,
    ErrorCode.CAPABILITY_DENIED,
    ErrorCode.COMMAND_DENIED,
    ErrorCode.WORKSPACE_ACCESS_DENIED,
}
_MAX_DECISION_REPAIR_ATTEMPTS = 1


class WorkerRuntime:
    def __init__(
        self,
        *,
        model_gateway: ModelGateway,
        tool_executor: ToolExecutor,
        context_builder: ContextBuilder,
        artifact_store: ArtifactStore,
        workspace_provider: WorkspaceProvider,
        event_recorder: EventRecorder,
        command_runner: CommandRunner | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._model_gateway = model_gateway
        self._tool_executor = tool_executor
        self._context_builder = context_builder
        self._artifact_store = artifact_store
        self._workspace_provider = workspace_provider
        self._event_recorder = event_recorder
        self._command_runner = command_runner
        self._clock = clock
        self._result_builder = ResultBuilder()

    def execute(self, execution_spec: AgentExecutionSpec) -> TaskResult:
        tracker = RuntimeLimitTracker(execution_spec.runtime_limits, clock=self._clock)
        self._record(
            execution_spec,
            EventType.AGENT_STARTED,
            {
                "agent_id": str(execution_spec.agent_id),
                "execution_spec_id": str(execution_spec.spec_id),
                "execution_spec_version": execution_spec.version,
                "model_key": execution_spec.model_routing.selected_model_key,
                "tool_names": list(execution_spec.tool_names),
            },
        )
        try:
            workspace = self._workspace_provider.for_execution(execution_spec)
            bundle = self._context_builder.build(execution_spec)
            estimated_input_tokens = max(1, (len(bundle.text) + 3) // 4)
            if estimated_input_tokens > execution_spec.runtime_limits.max_input_tokens:
                raise DomainError(
                    ErrorCode.CONTEXT_LIMIT_EXCEEDED,
                    "初始文本上下文估算 Token 超过执行规格上限",
                )
            messages = [_SYSTEM_MESSAGE, bundle.text]
            tool_contracts = [
                definition.model_dump(mode="json")
                for definition in self._tool_executor.definitions_for(execution_spec.tool_names)
            ]
            messages.append(
                "本实例可用能力协议："
                + json.dumps(tool_contracts, ensure_ascii=False, sort_keys=True)
            )
            messages.append(
                "Worker 决策 JSON Schema："
                + json.dumps(worker_decision_json_schema(), ensure_ascii=False, sort_keys=True)
            )
            evidence: list[EvidenceRef] = []
            artifacts: list[ArtifactRef] = []
            decision_repair_attempts = 0
            while True:
                tracker.record_step()
                _assert_message_estimate(tuple(messages), execution_spec)
                response = self._complete(execution_spec, tuple(messages))
                tracker.record_model_response(response)
                self._record(
                    execution_spec,
                    EventType.MODEL_CALL_COMPLETED,
                    {
                        "model_key": response.model_key,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "reasoning_tokens": response.reasoning_tokens,
                    },
                )
                try:
                    decision = parse_worker_decision(response.text)
                except DomainError as error:
                    if (
                        error.code is not ErrorCode.INVALID_MODEL_OUTPUT
                        or decision_repair_attempts >= _MAX_DECISION_REPAIR_ATTEMPTS
                    ):
                        raise
                    decision_repair_attempts += 1
                    self._record(
                        execution_spec,
                        EventType.MODEL_OUTPUT_REJECTED,
                        {
                            "error_code": error.code.value,
                            "repair_attempt": decision_repair_attempts,
                        },
                    )
                    messages.append(
                        "协议拒绝观察："
                        + json.dumps(
                            {
                                "error_code": error.code.value,
                                "instruction": (
                                    "上次响应没有执行。请只返回一个符合已提供 JSON Schema 的"
                                    "tool 或 finish 对象；不要添加第二个对象或私有推理。"
                                ),
                                "remaining_repair_attempts": (
                                    _MAX_DECISION_REPAIR_ATTEMPTS - decision_repair_attempts
                                ),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    continue
                if isinstance(decision, FinishDecision):
                    result = self._result_builder.build(
                        spec=execution_spec,
                        decision=decision,
                        evidence=evidence,
                        artifacts=artifacts,
                        usage=tracker.usage(),
                    )
                    self._record_finished(execution_spec, result)
                    return result
                if isinstance(decision, ToolCallDecision):
                    tracker.record_tool_call()
                    argument_keys: list[JsonValue] = []
                    for key in sorted(decision.arguments):
                        argument_keys.append(key)
                    self._record(
                        execution_spec,
                        EventType.TOOL_STARTED,
                        {
                            "tool_name": decision.tool_name,
                            "argument_keys": argument_keys,
                        },
                    )
                    try:
                        capability_result = self._tool_executor.execute(
                            execution_spec=execution_spec,
                            tool_name=decision.tool_name,
                            arguments=decision.arguments,
                            workspace=workspace,
                            artifact_store=self._artifact_store,
                            command_runner=self._command_runner,
                        )
                    except DomainError as error:
                        self._record(
                            execution_spec,
                            EventType.TOOL_REJECTED,
                            {"tool_name": decision.tool_name, "error_code": error.code.value},
                        )
                        if error.code not in _RECOVERABLE_TOOL_ERRORS:
                            raise
                        messages.append(
                            "工具拒绝观察："
                            + json.dumps(
                                {
                                    "tool_name": decision.tool_name,
                                    "error_code": error.code.value,
                                    "message": error.message,
                                    "details": error.details,
                                    "instruction": (
                                        "该操作没有执行。请在现有能力和路径范围内修正参数；"
                                        "不要重复同一个被拒绝的调用。"
                                    ),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            )
                        )
                        continue
                    evidence.extend(capability_result.evidence)
                    artifacts.extend(capability_result.artifacts)
                    for artifact in capability_result.artifacts:
                        self._record(
                            execution_spec,
                            EventType.ARTIFACT_PERSISTED,
                            {
                                "artifact_id": str(artifact.artifact_id),
                                "kind": artifact.kind.value,
                                "sha256": artifact.sha256,
                            },
                        )
                    messages.append(
                        "工具观察："
                        + json.dumps(
                            {
                                "tool_name": decision.tool_name,
                                "summary": capability_result.summary,
                                "details": capability_result.details,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    if (
                        decision.tool_name == "file.write"
                        and capability_result.details.get("changed") is False
                    ):
                        messages.append(
                            "收尾约束：文件内容没有变化。不要再次提交相同写入；"
                            "若验收条件已满足，立即输出 finish。"
                        )
                    evidence_ids: list[JsonValue] = [
                        str(item.evidence_id) for item in capability_result.evidence
                    ]
                    artifact_ids: list[JsonValue] = [
                        str(item.artifact_id) for item in capability_result.artifacts
                    ]
                    self._record(
                        execution_spec,
                        EventType.TOOL_FINISHED,
                        {
                            "tool_name": decision.tool_name,
                            "summary": capability_result.summary,
                            "evidence_ids": evidence_ids,
                            "artifact_ids": artifact_ids,
                        },
                    )
        except DomainError as domain_error:
            result = build_failure_result(
                spec=execution_spec,
                error=domain_error,
                usage=tracker.usage(),
            )
            self._record_finished(execution_spec, result)
            return result

    def _complete(
        self,
        spec: AgentExecutionSpec,
        messages: tuple[str, ...],
    ) -> ModelResponse:
        request = ModelRequest(
            model_key=spec.model_routing.selected_model_key,
            messages=messages,
            max_output_tokens=spec.runtime_limits.max_output_tokens,
            reasoning_effort=spec.model_routing.reasoning_effort,
        )
        try:
            response = self._model_gateway.complete(request)
        except DomainError:
            raise
        except Exception as exc:
            raise DomainError(
                ErrorCode.MODEL_UNAVAILABLE,
                "模型网关调用失败",
                details={"exception_type": type(exc).__name__},
            ) from exc
        if response.model_key != spec.model_routing.selected_model_key:
            raise DomainError(
                ErrorCode.MODEL_CAPABILITY_MISMATCH,
                "模型响应与执行规格选择的模型不一致",
            )
        return response

    def _record(
        self,
        spec: AgentExecutionSpec,
        event_type: EventType,
        payload: dict[str, JsonValue],
    ) -> None:
        self._event_recorder.record(
            run_id=spec.run_id,
            task_id=spec.task_id,
            event_type=event_type,
            payload=payload,
        )

    def _record_finished(self, spec: AgentExecutionSpec, result: TaskResult) -> None:
        self._record(
            spec,
            EventType.AGENT_FINISHED,
            {
                "agent_id": str(spec.agent_id),
                "status": result.status.value,
                "error_code": result.failure.code.value if result.failure else None,
                "evidence_count": len(result.evidence),
                "artifact_count": len(result.artifacts),
            },
        )


def _assert_message_estimate(messages: tuple[str, ...], spec: AgentExecutionSpec) -> None:
    estimated_tokens = max(1, (sum(len(message) for message in messages) + 3) // 4)
    if estimated_tokens > spec.runtime_limits.max_input_tokens:
        raise DomainError(
            ErrorCode.CONTEXT_LIMIT_EXCEEDED,
            "累计 Worker 消息估算 Token 超过执行规格上限",
        )
