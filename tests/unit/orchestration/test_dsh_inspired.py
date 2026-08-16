"""Tests for dsh-inspired mechanisms: continuable workers, transactional
factory, evidence.read, and calibrated difficulty."""

from __future__ import annotations

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.enums import InputModality, ModelAvailability
from evoweave_ds.domain.identifiers import AgentId, RunId, SpecId, TaskId
from evoweave_ds.domain.model_routing import ModelRoutingDecision
from evoweave_ds.orchestration.agent_factory import InProcessWorkerProvider
from evoweave_ds.repository.impact_analysis import _semantic_difficulty_text


class TestSemanticDifficulty:
    def test_behavior_terms_raise_difficulty(self) -> None:
        result = _semantic_difficulty_text("转写失败自动重试一次，仍失败则回退")
        assert result.score >= 2
        assert any("行为逻辑词" in reason for reason in result.reasons)

    def test_compat_terms_lower_difficulty(self) -> None:
        result = _semantic_difficulty_text("增加可选参数，未提供时行为保持不变")
        assert result.score <= -3

    def test_plain_text_neutral(self) -> None:
        result = _semantic_difficulty_text("修改一个默认值")
        assert result.score == 0
        assert result.reasons == ()


class TestWorkerProvider:
    def test_in_process_provider_register_unregister(self) -> None:
        provider = InProcessWorkerProvider()
        assert provider.name == "in-process"
        spec = _stub_spec()
        provider.register(spec)
        assert provider.get(spec.spec_id) is not None
        provider.unregister(spec.spec_id)
        assert provider.get(spec.spec_id) is None


def _stub_spec() -> AgentExecutionSpec:
    return AgentExecutionSpec(
        spec_id=SpecId.new(),
        run_id=RunId.new(),
        agent_id=AgentId.new(),
        task_id=TaskId.new(),
        task_spec_id=SpecId.new(),
        task_spec_version=1,
        base_commit="c40a5a78bc3e0b549932129713ac65806815c579",
        goal="测试目标",
        acceptance_criteria=("验收",),
        required_modalities=(InputModality.TEXT,),
        model_routing=ModelRoutingDecision(
            decision_id=SpecId.new(),
            requirement_id=SpecId.new(),
            requirement_version=1,
            selected_model_key="deepseek:deepseek-v4-flash",
            selected_availability=ModelAvailability.AVAILABLE,
            reason="测试",
        ),
        continuable=False,
    )
