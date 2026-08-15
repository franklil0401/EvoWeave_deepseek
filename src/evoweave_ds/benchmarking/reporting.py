"""Write explicit benchmark reports and leave missing evidence visibly pending."""

import json
import os
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from evoweave_ds.benchmarking.metrics import StrategyMetrics, aggregate_metrics
from evoweave_ds.benchmarking.models import (
    AgentStrategy,
    BenchmarkRunRecord,
    BenchmarkSuite,
    EvidenceLevel,
    ModelStrategy,
)
from evoweave_ds.domain.base import DomainModel


class GoNoGoStatus(StrEnum):
    GO = "go"
    NO_GO = "no_go"
    PENDING = "pending"


class GoNoGoAssessment(DomainModel):
    status: GoNoGoStatus
    evidence_level: EvidenceLevel | None = None
    reasons: tuple[str, ...] = Field(min_length=1)


class BenchmarkReportWriter:
    def write(
        self,
        *,
        suite: BenchmarkSuite,
        suite_sha256: str,
        records: tuple[BenchmarkRunRecord, ...],
        output_root: Path | str,
    ) -> tuple[Path, Path]:
        mismatched = {item.suite_sha256 for item in records if item.suite_sha256 != suite_sha256}
        if mismatched:
            raise ValueError("benchmark 运行记录与当前任务集 SHA-256 不一致")
        system_commits = {item.system_commit for item in records}
        if len(system_commits) > 1:
            raise ValueError("benchmark 运行记录混入了多个系统 Git 提交")
        metrics = aggregate_metrics(suite, records)
        assessment = assess_go_no_go(suite, records, metrics)
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        markdown_path = root / "评测汇总报告.md"
        json_path = root / "评测汇总报告.json"
        _atomic_text(
            markdown_path,
            _markdown(
                suite,
                suite_sha256,
                metrics,
                assessment,
                system_commit=next(iter(system_commits), None),
            ),
        )
        payload = {
            "suite_id": suite.suite_id,
            "suite_version": suite.version,
            "suite_sha256": suite_sha256,
            "task_count": len(suite.tasks),
            "record_count": len(records),
            "system_commits": sorted(system_commits),
            "metrics": [item.model_dump(mode="json") for item in metrics],
            "go_no_go": assessment.model_dump(mode="json"),
        }
        _atomic_text(
            json_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        return markdown_path, json_path


def assess_go_no_go(
    suite: BenchmarkSuite,
    records: tuple[BenchmarkRunRecord, ...],
    metrics: tuple[StrategyMetrics, ...],
) -> GoNoGoAssessment:
    expected_per_group = len(suite.tasks)
    for level in (EvidenceLevel.LIVE_MODEL, EvidenceLevel.OFFLINE_REPLAY):
        level_metrics = tuple(item for item in metrics if item.evidence_level is level)
        if not _has_complete_trials(
            suite,
            records,
            evidence_level=level,
            expected_per_group=expected_per_group,
        ):
            continue
        dynamic = next(
            item
            for item in level_metrics
            if item.agent_strategy is AgentStrategy.ADAPTIVE
            and item.model_strategy is ModelStrategy.ADAPTIVE
        )
        baselines = tuple(item for item in level_metrics if item is not dynamic)
        best = max(baselines, key=lambda item: item.success_rate)
        performance = dynamic.success_rate >= best.success_rate - 0.05 and (
            dynamic.average_tokens <= best.average_tokens * 0.8
            or dynamic.average_duration_ms <= best.average_duration_ms * 0.8
        )
        simple_ids = {task.benchmark_id for task in suite.tasks if "simple" in task.scenario_tags}
        simple_dynamic = tuple(
            record
            for record in records
            if record.evidence_level is level
            and record.agent_strategy is AgentStrategy.ADAPTIVE
            and record.model_strategy is ModelStrategy.ADAPTIVE
            and record.benchmark_id in simple_ids
        )
        simple_minimal = bool(simple_dynamic) and (
            sum(record.agent_count == 1 for record in simple_dynamic) / len(simple_dynamic) >= 0.8
        )
        route_hard_constraints = dynamic.route_hard_constraint_compliance_rate == 1.0
        route_hard_constraint_reason = (
            "模型硬约束合规门槛：数据缺失"
            if dynamic.route_hard_constraint_compliance_rate is None
            else f"模型硬约束合规门槛：{'通过' if route_hard_constraints else '未通过'}"
        )
        context_reduced = dynamic.orchestrator_context_ratio <= 0.5
        passed = performance and simple_minimal and route_hard_constraints and context_reduced
        return GoNoGoAssessment(
            status=GoNoGoStatus.GO if passed else GoNoGoStatus.NO_GO,
            evidence_level=level,
            reasons=(
                f"性能门槛：{'通过' if performance else '未通过'}",
                f"简单任务最小实例门槛：{'通过' if simple_minimal else '未通过'}",
                route_hard_constraint_reason,
                f"调度上下文压缩门槛：{'通过' if context_reduced else '未通过'}",
            ),
        )
    return GoNoGoAssessment(
        status=GoNoGoStatus.PENDING,
        reasons=("尚无任一真实性等级完成 12 个任务 × 3 种 Agent 策略 × 3 种模型策略的完整矩阵",),
    )


def load_run_records(path: Path | str) -> tuple[BenchmarkRunRecord, ...]:
    payload = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("benchmark 结果文件顶层必须是数组")
    return tuple(BenchmarkRunRecord.model_validate(item) for item in payload)


def _markdown(
    suite: BenchmarkSuite,
    suite_sha256: str,
    metrics: tuple[StrategyMetrics, ...],
    assessment: GoNoGoAssessment,
    *,
    system_commit: str | None,
) -> str:
    lines = [
        "# EvoWeave 评测汇总报告",
        "",
        f"- 任务集：`{suite.suite_id}` v{suite.version}",
        f"- 任务集 SHA-256：`{suite_sha256}`",
        f"- 固定任务数：{len(suite.tasks)}",
        f"- 系统 Git 提交：`{system_commit or '无'}`",
        f"- Go/No-Go：`{assessment.status.value}`",
        f"- 证据等级：`{assessment.evidence_level.value if assessment.evidence_level else '无'}`",
        "",
        "## 判断依据",
        "",
        *(f"- {reason}" for reason in assessment.reasons),
        "",
        "## 策略指标",
        "",
    ]
    if not metrics:
        lines.append("尚无运行记录。本文档只证明任务集可复现，不代表任何策略已经胜出。")
    else:
        lines.extend(
            [
                "| Agent 策略 | 模型策略 | 证据 | 数量 | 成功率 | "
                "重复数 | 成功率标准差 | 平均 Token | 平均时延(ms) | 平均 Agent | "
                "硬约束合规 | 首次执行成功 |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                *(
                    "| "
                    f"{item.agent_strategy.value} | {item.model_strategy.value} | "
                    f"{item.evidence_level.value} | {item.run_count} | "
                    f"{item.success_rate:.1%} | {item.trial_count} | "
                    f"{item.success_rate_stddev:.1%} | {item.average_tokens:.1f} | "
                    f"{item.average_duration_ms:.1f} | {item.average_agent_count:.2f} | "
                    f"{_optional_percent(item.route_hard_constraint_compliance_rate)} | "
                    f"{item.initial_execution_success_rate:.1%} |"
                    for item in metrics
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 真实性说明",
            "",
            "`offline_replay` 只用于验证指标管线和确定性行为；只有 `live_model` "
            "可用于简历中的模型效果数字。缺失组合不会被零值填充，也不会参与 Go/No-Go。",
            "",
        ]
    )
    return "\n".join(lines)


def _has_complete_trials(
    suite: BenchmarkSuite,
    records: tuple[BenchmarkRunRecord, ...],
    *,
    evidence_level: EvidenceLevel,
    expected_per_group: int,
) -> bool:
    expected_tasks = {item.benchmark_id for item in suite.tasks}
    expected_groups = {(agent, model) for agent in AgentStrategy for model in ModelStrategy}
    tasks_by_trial: dict[tuple[AgentStrategy, ModelStrategy, int], set[str]] = {}
    for record in records:
        if record.evidence_level is not evidence_level:
            continue
        key = (record.agent_strategy, record.model_strategy, record.trial_index)
        tasks_by_trial.setdefault(key, set()).add(record.benchmark_id)
    trial_sets = {
        group: {trial for agent, model, trial in tasks_by_trial if (agent, model) == group}
        for group in expected_groups
    }
    common_trials = next(iter(trial_sets.values()), set())
    if not common_trials or any(trials != common_trials for trials in trial_sets.values()):
        return False
    return all(
        len(tasks_by_trial.get((agent, model, trial), set())) == expected_per_group
        and tasks_by_trial[(agent, model, trial)] == expected_tasks
        for agent, model in expected_groups
        for trial in common_trials
    )


def _optional_percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "N/A"


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
