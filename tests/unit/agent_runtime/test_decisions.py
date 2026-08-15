import json

import pytest

from evoweave_ds.agent_runtime.decisions import FinishDecision, parse_worker_decision
from evoweave_ds.domain.errors import DomainError, ErrorCode


def _finish_json() -> str:
    return json.dumps(
        {"action": "finish", "status": "succeeded", "summary": "任务完成"},
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    "response",
    [
        "```json\n" + _finish_json() + "\n```",
        "下面是结构化结果：\n" + _finish_json(),
        "结果：\n```json\n" + _finish_json() + "\n```",
        _finish_json() + "\n任务已经完成。",
        "```json\n" + _finish_json() + "\n```\n任务已经完成。",
        '{"note":"以下才是决策"}\n' + _finish_json(),
    ],
)
def test_parser_recovers_one_wrapped_json_object(response: str) -> None:
    assert isinstance(parse_worker_decision(response), FinishDecision)


@pytest.mark.parametrize(
    "response",
    [
        _finish_json() + "\n" + _finish_json(),
        "没有对象",
    ],
)
def test_parser_rejects_ambiguous_or_missing_payload(response: str) -> None:
    with pytest.raises(DomainError) as error:
        parse_worker_decision(response)
    assert error.value.code is ErrorCode.INVALID_MODEL_OUTPUT
