"""Finite structured decisions accepted from the model loop."""

import json
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


def parse_worker_decision(text: str) -> WorkerDecision:
    try:
        return _DECISION_ADAPTER.validate_json(text)
    except ValidationError as direct_error:
        return _extract_unique_worker_decision(text, direct_error=direct_error)


def _extract_unique_worker_decision(
    text: str,
    *,
    direct_error: ValidationError,
) -> WorkerDecision:
    """Recover exactly one schema-valid decision from prose without guessing fields."""

    valid: list[WorkerDecision] = []
    candidate_errors: list[dict[str, object]] = []
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            payload, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as error:
            if len(candidate_errors) < 8:
                candidate_errors.append(
                    {"type": type(error).__name__, "msg": str(error), "offset": start}
                )
            cursor = start + 1
            continue
        cursor = start + max(consumed, 1)
        if not isinstance(payload, dict):
            continue
        try:
            valid.append(_DECISION_ADAPTER.validate_python(payload))
        except ValidationError as error:
            if len(candidate_errors) < 8:
                candidate_errors.append(
                    {
                        "type": "schema_validation",
                        "offset": start,
                        "errors": error.errors(include_url=False),
                    }
                )

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
            "candidate_errors": candidate_errors,
        },
    )


def worker_decision_json_schema() -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], _DECISION_ADAPTER.json_schema())
