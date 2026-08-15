"""Finite structured decisions accepted from the model loop."""

import json
import re
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import ResultStatus, RiskLevel
from evoweave_ds.domain.errors import DomainError, ErrorCode


class ToolCallDecision(DomainModel):
    action: Literal["tool"]
    tool_name: str = Field(min_length=3, max_length=128)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class FinishDecision(DomainModel):
    action: Literal["finish"]
    status: ResultStatus
    summary: str = Field(min_length=1, max_length=10_000)
    risk_level: RiskLevel = RiskLevel.LOW
    risk_notes: tuple[str, ...] = ()
    failure_code: ErrorCode | None = None
    failure_message: str | None = Field(default=None, min_length=1, max_length=2_000)
    retryable: bool = False

    @model_validator(mode="after")
    def validate_finish_state(self) -> "FinishDecision":
        if self.status is ResultStatus.SUCCEEDED:
            if self.failure_code is not None or self.failure_message is not None:
                raise ValueError("成功决定不能包含失败信息")
        elif self.failure_code is None or self.failure_message is None:
            raise ValueError("非成功决定必须包含失败代码和消息")
        return self


WorkerDecision = Annotated[ToolCallDecision | FinishDecision, Field(discriminator="action")]
_DECISION_ADAPTER: TypeAdapter[WorkerDecision] = TypeAdapter(WorkerDecision)

_THOUGHT_PATTERN = re.compile(r"<thought>.*?</thought>", re.DOTALL)
_INVOKE_PATTERN = re.compile(r"<invoke name=\"([A-Za-z0-9_.-]+)\">(.*?)</invoke>", re.DOTALL)
_PARAMETER_PATTERN = re.compile(r"<parameter name=\"([^\"]+)\">(.*?)</parameter>", re.DOTALL)


def parse_worker_decision(text: str) -> WorkerDecision:
    """Strict single-decision parsing used by tests and strict callers."""

    try:
        return _DECISION_ADAPTER.validate_json(text)
    except ValidationError as direct_error:
        xml_decisions = _parse_anthropic_tool_calls(text)
        if xml_decisions is not None and len(xml_decisions) == 1:
            return xml_decisions[0]
        return _extract_unique_worker_decision(text, direct_error=direct_error)


def parse_worker_decisions(text: str) -> tuple[WorkerDecision, ...]:
    """Parse an ordered sequence of decisions (JSON or Anthropic XML).

    Real models frequently emit several tool calls in one response. The worker
    loop executes them in order; a finish decision stops the sequence.
    """

    try:
        return (_DECISION_ADAPTER.validate_json(text),)
    except ValidationError as direct_error:
        xml_decisions = _parse_anthropic_tool_calls(text)
        if xml_decisions is not None:
            return xml_decisions
        decisions = _extract_all_worker_decisions(text, direct_error=direct_error)
        if not decisions:
            raise DomainError(
                ErrorCode.INVALID_MODEL_OUTPUT,
                "模型输出不符合 Worker 决策协议",
                details={
                    "direct_errors": direct_error.errors(include_url=False),
                },
            ) from direct_error
        return decisions


def _parse_anthropic_tool_calls(text: str) -> tuple[ToolCallDecision, ...] | None:
    """Accept Anthropic-style <invoke> blocks as tool decisions in document order.

    deepseek-v4-flash sometimes emits Anthropic XML tool calls instead of JSON
    under long contexts. Parameter values are kept as strings and later
    validated by the capability argument schemas.
    """

    text = _THOUGHT_PATTERN.sub("", text)
    matches = _INVOKE_PATTERN.findall(text)
    if not matches:
        return None
    decisions: list[ToolCallDecision] = []
    for tool_name, body in matches:
        parameters = {key.strip(): value.strip() for key, value in _PARAMETER_PATTERN.findall(body)}
        if not tool_name:
            return None
        decisions.append(ToolCallDecision(action="tool", tool_name=tool_name, arguments=parameters))
    return tuple(decisions)


def _extract_unique_worker_decision(
    text: str,
    *,
    direct_error: ValidationError,
) -> WorkerDecision:
    valid = _extract_all_worker_decisions(text, direct_error=direct_error)
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        raise DomainError(
            ErrorCode.INVALID_MODEL_OUTPUT,
            "模型输出包含多个有效 Worker 决策，无法安全判定",
            details={"valid_candidate_count": len(valid)},
        )
    raise DomainError(
        ErrorCode.INVALID_MODEL_OUTPUT,
        "模型输出不符合 Worker 决策协议",
        details={
            "direct_errors": direct_error.errors(include_url=False),
        },
    )


def _extract_all_worker_decisions(
    text: str,
    *,
    direct_error: ValidationError,
) -> tuple[WorkerDecision, ...]:
    """Collect every schema-valid decision from prose in document order.

    Real models occasionally attach fields from the other decision branch (e.g. a
    status field on a tool object). We tolerate that by retrying strict candidates
    after dropping protocol-external keys, without ever guessing missing values.
    """

    valid: list[WorkerDecision] = []
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            payload, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + max(consumed, 1)
        if not isinstance(payload, dict):
            continue
        decision = _validate_decision(payload)
        if decision is None:
            coerced = _drop_external_keys(payload)
            if coerced is not None:
                decision = _validate_decision(coerced)
        if decision is not None:
            valid.append(decision)
    return tuple(valid)


def _validate_decision(payload: dict[str, JsonValue]) -> WorkerDecision | None:
    try:
        return _DECISION_ADAPTER.validate_python(payload)
    except ValidationError:
        return None


_TOOL_KEYS = frozenset({"action", "tool_name", "arguments"})
_FINISH_KEYS = frozenset(
    {
        "action",
        "status",
        "summary",
        "risk_level",
        "risk_notes",
        "failure_code",
        "failure_message",
        "retryable",
    }
)


def _drop_external_keys(payload: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    action = payload.get("action")
    if action == "tool":
        return {key: value for key, value in payload.items() if key in _TOOL_KEYS}
    if action == "finish":
        return {key: value for key, value in payload.items() if key in _FINISH_KEYS}
    return None


def worker_decision_json_schema() -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], _DECISION_ADAPTER.json_schema())
