"""Tests for minimal text-only context construction."""

import pytest

from evoweave_ds.agent_runtime.context_builder import ContextBuilder, ContextPolicy
from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import ArtifactKind, InputModality
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelRoutingDecision
from evoweave_ds.infrastructure.artifacts.memory import InMemoryArtifactStore


def _spec(
    *,
    task_id: TaskId,
    input_ids: tuple = (),
    context_ids: tuple = (),
) -> AgentExecutionSpec:
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=task_id,
        task_spec_id=SpecId.new(),
        task_spec_version=1,
        base_commit="a" * 40,
        goal="理解界面并更新代码",
        acceptance_criteria=("界面符合需求",),
        required_modalities=(InputModality.TEXT,),
        model_routing=ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=SpecId.new(),
            requirement_version=1,
            selected_model_key="fake:flash",
            reasoning_effort="low",
            reason="测试",
        ),
        read_scope=("src",),
        input_artifact_ids=input_ids,
        context_artifact_ids=context_ids,
    )


def test_context_includes_goal_acceptance_and_capabilities() -> None:
    store = InMemoryArtifactStore()
    bundle = ContextBuilder(store).build(_spec(task_id=TaskId.new()))
    assert "理解界面并更新代码" in bundle.text
    assert "验收条件" in bundle.text
    assert bundle.included_artifact_ids == ()


def test_context_inlines_text_artifact_content() -> None:
    store = InMemoryArtifactStore()
    context = store.put_bytes(
        b"repository profile snapshot",
        media_type="application/json",
        kind=ArtifactKind.REPOSITORY_PROFILE,
    )
    bundle = ContextBuilder(store).build(
        _spec(task_id=TaskId.new(), context_ids=(context.artifact_id,))
    )
    assert "repository profile snapshot" in bundle.text
    assert bundle.included_artifact_ids == (context.artifact_id,)


def test_context_references_binary_artifact_without_inlining() -> None:
    store = InMemoryArtifactStore()
    binary = store.put_bytes(
        b"\x00\x01binary",
        media_type="application/octet-stream",
        kind=ArtifactKind.GENERIC,
    )
    bundle = ContextBuilder(store).build(
        _spec(task_id=TaskId.new(), context_ids=(binary.artifact_id,))
    )
    assert str(binary.artifact_id) in bundle.text
    assert "二进制产物引用" in bundle.text


def test_context_byte_limit_rejects_large_artifact() -> None:
    store = InMemoryArtifactStore()
    context = store.put_bytes(
        b"large context",
        media_type="text/plain",
        kind=ArtifactKind.CONTEXT_BUNDLE,
    )
    with pytest.raises(DomainError) as error:
        ContextBuilder(store, ContextPolicy(max_artifact_bytes=2)).build(
            _spec(
                task_id=TaskId.new(),
                context_ids=(context.artifact_id,),
            )
        )
    assert error.value.code is ErrorCode.CONTEXT_LIMIT_EXCEEDED
