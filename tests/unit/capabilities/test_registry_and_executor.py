"""Tests for capability registration, grants, scopes, and command policy."""

import pytest

from evoweave_ds.capabilities.builtins import (
    CommandRunCapability,
    FileReadCapability,
    FileWriteCapability,
)
from evoweave_ds.capabilities.command_policy import CommandPolicy
from evoweave_ds.capabilities.registry import CapabilityRegistry
from evoweave_ds.capabilities.tool_executor import ToolExecutor
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelRoutingDecision
from evoweave_ds.domain.ports import CommandResult
from evoweave_ds.infrastructure.artifacts.memory import InMemoryArtifactStore
from evoweave_ds.infrastructure.commands.fake import ScriptedCommandRunner
from evoweave_ds.infrastructure.workspaces.fake import FakeWorkspace


def _spec(
    task_id: TaskId,
    *,
    tools: tuple[str, ...],
    write_scope: tuple[str, ...] = (),
    allowed_commands: tuple[str, ...] = (),
) -> AgentExecutionSpec:
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=task_id,
        task_spec_id=SpecId.new(),
        task_spec_version=1,
        base_commit="a" * 40,
        goal="执行原子能力测试",
        acceptance_criteria=("行为符合权限",),
        model_routing=ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=SpecId.new(),
            requirement_version=1,
            selected_model_key="fake:model",
            reason="测试",
        ),
        tool_names=tools,
        allowed_commands=allowed_commands,
        read_scope=("src",),
        write_scope=write_scope,
    )


def test_registry_rejects_duplicate_capability_name() -> None:
    registry = CapabilityRegistry((FileReadCapability(),))
    with pytest.raises(DomainError) as error:
        registry.register(FileReadCapability())
    assert error.value.code is ErrorCode.INVALID_SPEC


def test_registered_capability_is_not_implicitly_granted() -> None:
    task_id = TaskId.new()
    workspace = FakeWorkspace(
        task_id=task_id,
        files={"src/app.py": "value = 1\n"},
        read_scope=("src",),
    )
    executor = ToolExecutor(CapabilityRegistry((FileReadCapability(),)))
    with pytest.raises(DomainError) as error:
        executor.execute(
            execution_spec=_spec(task_id, tools=()),
            tool_name="file.read",
            arguments={"path": "src/app.py"},
            workspace=workspace,
            artifact_store=InMemoryArtifactStore(),
        )
    assert error.value.code is ErrorCode.CAPABILITY_DENIED


def test_write_capability_requires_write_scope() -> None:
    task_id = TaskId.new()
    executor = ToolExecutor(CapabilityRegistry((FileWriteCapability(),)))
    with pytest.raises(DomainError) as error:
        executor.execute(
            execution_spec=_spec(task_id, tools=("file.write",)),
            tool_name="file.write",
            arguments={"path": "src/app.py", "content": "new\n"},
            workspace=FakeWorkspace(
                task_id=task_id,
                files={"src/app.py": "old\n"},
                read_scope=("src",),
            ),
            artifact_store=InMemoryArtifactStore(),
        )
    assert error.value.code is ErrorCode.CAPABILITY_DENIED


def test_workspace_still_enforces_path_after_tool_grant() -> None:
    task_id = TaskId.new()
    executor = ToolExecutor(CapabilityRegistry((FileReadCapability(),)))
    with pytest.raises(DomainError) as error:
        executor.execute(
            execution_spec=_spec(task_id, tools=("file.read",)),
            tool_name="file.read",
            arguments={"path": "secrets.txt"},
            workspace=FakeWorkspace(
                task_id=task_id,
                files={"secrets.txt": "hidden"},
                read_scope=("src",),
            ),
            artifact_store=InMemoryArtifactStore(),
        )
    assert error.value.code is ErrorCode.WORKSPACE_ACCESS_DENIED


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-c", "print(1)"],
        ["pytest", "tests && whoami"],
        ["tools/pytest", "-q"],
    ],
)
def test_command_policy_rejects_unlisted_or_shell_controlled_command(argv: list[str]) -> None:
    task_id = TaskId.new()
    runner = ScriptedCommandRunner()
    executor = ToolExecutor(
        CapabilityRegistry((CommandRunCapability(),)),
        command_policy=CommandPolicy(),
    )
    with pytest.raises(DomainError) as error:
        executor.execute(
            execution_spec=_spec(
                task_id,
                tools=("command.run",),
                allowed_commands=("pytest",),
            ),
            tool_name="command.run",
            arguments={"argv": argv},
            workspace=FakeWorkspace(task_id=task_id, read_scope=("src",)),
            artifact_store=InMemoryArtifactStore(),
            command_runner=runner,
        )
    assert error.value.code is ErrorCode.COMMAND_DENIED
    assert not runner.calls


def test_authorized_command_runs_without_shell() -> None:
    task_id = TaskId.new()
    argv = ("pytest", "-q")
    runner = ScriptedCommandRunner({argv: CommandResult(argv=argv, exit_code=0, stdout="1 passed")})
    executor = ToolExecutor(CapabilityRegistry((CommandRunCapability(),)))
    result = executor.execute(
        execution_spec=_spec(
            task_id,
            tools=("command.run",),
            allowed_commands=("pytest",),
        ),
        tool_name="command.run",
        arguments={"argv": list(argv)},
        workspace=FakeWorkspace(task_id=task_id, read_scope=("src",)),
        artifact_store=InMemoryArtifactStore(),
        command_runner=runner,
    )
    assert result.details["exit_code"] == 0
    assert runner.calls == [(argv, 120)]
