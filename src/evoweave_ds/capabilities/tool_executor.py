"""Programmatic grant checks in front of every capability invocation."""

from pydantic import JsonValue

from evoweave_ds.capabilities.command_policy import CommandPolicy
from evoweave_ds.capabilities.definitions import (
    CapabilityContext,
    CapabilityDefinition,
    CapabilityResult,
)
from evoweave_ds.capabilities.registry import CapabilityRegistry
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import CapabilityAccess
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.ports import ArtifactStore, CommandRunner, WorkspaceAdapter


class ToolExecutor:
    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        command_policy: CommandPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._command_policy = command_policy or CommandPolicy()

    def definitions_for(self, tool_names: tuple[str, ...]) -> tuple[CapabilityDefinition, ...]:
        return tuple(self._registry.get(name).definition for name in tool_names)

    def execute(
        self,
        *,
        execution_spec: AgentExecutionSpec,
        tool_name: str,
        arguments: dict[str, JsonValue],
        workspace: WorkspaceAdapter,
        artifact_store: ArtifactStore,
        command_runner: CommandRunner | None = None,
    ) -> CapabilityResult:
        capability = self._registry.get(tool_name)
        if tool_name not in execution_spec.tool_names:
            raise DomainError(
                ErrorCode.CAPABILITY_DENIED,
                f"能力未被执行规格授予：{tool_name}",
            )
        access = capability.definition.access
        if access is CapabilityAccess.READ and not execution_spec.read_scope:
            raise DomainError(ErrorCode.CAPABILITY_DENIED, "执行规格没有读取范围")
        if access is CapabilityAccess.WRITE and not execution_spec.write_scope:
            raise DomainError(ErrorCode.CAPABILITY_DENIED, "执行规格没有写入范围")
        if access is CapabilityAccess.COMMAND:
            argv = _parse_argv(arguments)
            self._command_policy.authorize(
                argv,
                allowed_commands=execution_spec.allowed_commands,
            )
            if command_runner is None:
                raise DomainError(ErrorCode.COMMAND_DENIED, "未配置命令执行适配器")

        context = CapabilityContext(
            execution_spec=execution_spec,
            workspace=workspace,
            artifact_store=artifact_store,
            command_runner=command_runner,
        )
        return capability.invoke(arguments, context)


def _parse_argv(arguments: dict[str, JsonValue]) -> tuple[str, ...]:
    raw = arguments.get("argv")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise DomainError(ErrorCode.COMMAND_DENIED, "命令能力必须提供字符串 argv 数组")
    return tuple(item for item in raw if isinstance(item, str))
