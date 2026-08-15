import json

import pytest

from evoweave_ds.agent_runtime.decisions import (
    FinishDecision,
    ToolCallDecision,
    parse_worker_decision,
    parse_worker_decisions,
)
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


def test_parser_tolerates_foreign_fields_on_tool_object() -> None:
    response = json.dumps(
        {
            "action": "tool",
            "tool_name": "file.read",
            "arguments": {"path": "src/app.py"},
            "status": "succeeded",
            "summary": "尝试读取文件",
        },
        ensure_ascii=False,
    )
    decision = parse_worker_decision(response)
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool_name == "file.read"
    assert decision.arguments == {"path": "src/app.py"}


def test_parser_tolerates_foreign_fields_on_finish_object() -> None:
    response = json.dumps(
        {
            "action": "finish",
            "status": "succeeded",
            "summary": "任务完成",
            "tool_name": "file.read",
        },
        ensure_ascii=False,
    )
    decision = parse_worker_decision(response)
    assert isinstance(decision, FinishDecision)
    assert decision.summary == "任务完成"


def test_parser_accepts_single_anthropic_xml_tool_call() -> None:
    response = (
        "<tool_calls>\n"
        '<invoke name="file.read">\n'
        '<parameter name="path">app/config.py</parameter>\n'
        "</invoke>\n"
        "</tool_calls>"
    )
    decision = parse_worker_decision(response)
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool_name == "file.read"
    assert decision.arguments == {"path": "app/config.py"}


def test_multi_parser_returns_multiple_anthropic_tool_calls_in_order() -> None:
    response = (
        "<tool_calls>\n"
        '<invoke name="file.read">\n<parameter name="path">a.py</parameter>\n</invoke>\n'
        '<invoke name="file.search">\n<parameter name="query">x</parameter>\n</invoke>\n'
        "</tool_calls>"
    )
    decisions = parse_worker_decisions(response)
    assert len(decisions) == 2
    assert isinstance(decisions[0], ToolCallDecision)
    assert decisions[0].tool_name == "file.read"
    assert isinstance(decisions[1], ToolCallDecision)
    assert decisions[1].tool_name == "file.search"


def test_multi_parser_returns_multiple_json_decisions_in_order() -> None:
    response = (
        '{"action": "tool", "tool_name": "file.read", "arguments": {"path": "a.py"}}\n'
        '{"action": "tool", "tool_name": "file.write", '
        '"arguments": {"path": "b.py", "content": "c"}}'
    )
    decisions = parse_worker_decisions(response)
    assert len(decisions) == 2
    assert [item.tool_name for item in decisions] == ["file.read", "file.write"]


def test_strict_parser_still_rejects_multiple_json_decisions() -> None:
    response = (
        '{"action": "tool", "tool_name": "file.read", "arguments": {"path": "a.py"}}\n'
        '{"action": "finish", "status": "succeeded", "summary": "done"}'
    )
    with pytest.raises(DomainError) as error:
        parse_worker_decision(response)
    assert error.value.code is ErrorCode.INVALID_MODEL_OUTPUT


def test_parser_strips_thought_block_before_xml_tool_call() -> None:
    response = (
        "<thought>我先读取文件了解结构。</thought>\n"
        "<tool_calls>\n"
        '<invoke name="file.read">\n<parameter name="path">app/config.py</parameter>\n</invoke>\n'
        "</tool_calls>"
    )
    decision = parse_worker_decision(response)
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool_name == "file.read"
    assert decision.arguments == {"path": "app/config.py"}


def test_parser_rejects_multiple_anthropic_tool_calls() -> None:
    response = (
        "<tool_calls>\n"
        '<invoke name="file.read">\n<parameter name="path">a.py</parameter>\n</invoke>\n'
        '<invoke name="file.write">\n<parameter name="path">b.py</parameter>\n</invoke>\n'
        "</tool_calls>"
    )
    with pytest.raises(DomainError) as error:
        parse_worker_decision(response)
    assert error.value.code is ErrorCode.INVALID_MODEL_OUTPUT
