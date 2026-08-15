"""Create a new immutable execution-spec version after explicit rerouting."""

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import AgentId, SpecId
from evoweave_ds.domain.model_routing import ModelRoutingDecision


def revise_execution_spec_for_routing(
    previous: AgentExecutionSpec,
    decision: ModelRoutingDecision,
) -> AgentExecutionSpec:
    if decision.requirement_id != previous.model_routing.requirement_id:
        raise DomainError(ErrorCode.INVALID_SPEC, "新路由决策不属于原模型需求")
    if decision.requirement_version != previous.model_routing.requirement_version:
        raise DomainError(ErrorCode.INVALID_SPEC, "新路由决策的需求版本发生变化")
    if decision.selected_model_key == previous.model_routing.selected_model_key:
        raise DomainError(ErrorCode.INVALID_SPEC, "回退必须选择不同模型")
    if decision.decision_id == previous.model_routing.decision_id:
        raise DomainError(ErrorCode.INVALID_SPEC, "回退必须生成新的路由决策")
    versioned_decision = decision.model_copy(
        update={"version": max(decision.version, previous.model_routing.version + 1)}
    )
    return previous.model_copy(
        update={
            "spec_id": SpecId.new(),
            "agent_id": AgentId.new(),
            "model_routing": versioned_decision,
            "version": previous.version + 1,
        }
    )
