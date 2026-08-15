"""Aggregate comparable metrics without inventing missing benchmark outcomes."""

from collections import defaultdict
from statistics import pstdev

from pydantic import Field

from evoweave_ds.benchmarking.models import (
    AgentStrategy,
    BenchmarkRunRecord,
    BenchmarkRunStatus,
    BenchmarkSuite,
    BenchmarkTask,
    EvidenceLevel,
    ModelStrategy,
)
from evoweave_ds.domain.base import DomainModel


class StrategyMetrics(DomainModel):
    agent_strategy: AgentStrategy
    model_strategy: ModelStrategy
    evidence_level: EvidenceLevel
    run_count: int = Field(ge=1)
    trial_count: int = Field(ge=1)
    success_rate: float = Field(ge=0.0, le=1.0)
    success_rate_stddev: float = Field(ge=0.0, le=1.0)
    localization_recall: float = Field(ge=0.0, le=1.0)
    patch_efficiency: float = Field(ge=0.0, le=1.0)
    regression_rate: float = Field(ge=0.0, le=1.0)
    average_tokens: float = Field(ge=0.0)
    average_tokens_stddev: float = Field(ge=0.0)
    average_duration_ms: float = Field(ge=0.0)
    average_duration_ms_stddev: float = Field(ge=0.0)
    average_agent_count: float = Field(ge=0.0)
    invalid_task_rate: float = Field(ge=0.0, le=1.0)
    conflict_rate: float = Field(ge=0.0, le=1.0)
    orchestrator_context_ratio: float = Field(ge=0.0)
    route_hard_constraint_compliance_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    initial_execution_success_rate: float = Field(ge=0.0, le=1.0)
    # Kept for v1 report compatibility; same meaning as initial_execution_success_rate.
    initial_route_success_rate: float = Field(ge=0.0, le=1.0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    difficulty_match_rate: float = Field(ge=0.0, le=1.0)


def aggregate_metrics(
    suite: BenchmarkSuite,
    records: tuple[BenchmarkRunRecord, ...],
) -> tuple[StrategyMetrics, ...]:
    task_by_id = {task.benchmark_id: task for task in suite.tasks}
    unknown = {record.benchmark_id for record in records}.difference(task_by_id)
    if unknown:
        raise ValueError(f"benchmark 结果引用未知任务：{sorted(unknown)}")
    duplicate_keys = [
        (
            record.benchmark_id,
            record.agent_strategy,
            record.model_strategy,
            record.evidence_level,
            record.trial_index,
        )
        for record in records
    ]
    if len(set(duplicate_keys)) != len(duplicate_keys):
        raise ValueError("同一任务和策略组合不能重复记录")
    grouped: defaultdict[
        tuple[AgentStrategy, ModelStrategy, EvidenceLevel],
        list[BenchmarkRunRecord],
    ] = defaultdict(list)
    for record in records:
        if record.status is not BenchmarkRunStatus.SKIPPED:
            grouped[(record.agent_strategy, record.model_strategy, record.evidence_level)].append(
                record
            )
    return tuple(
        _aggregate_group(task_by_id, key, tuple(group))
        for key, group in sorted(
            grouped.items(),
            key=lambda item: tuple(value.value for value in item[0]),
        )
    )


def _aggregate_group(
    task_by_id: dict[str, BenchmarkTask],
    key: tuple[AgentStrategy, ModelStrategy, EvidenceLevel],
    records: tuple[BenchmarkRunRecord, ...],
) -> StrategyMetrics:
    count = len(records)
    generated = [record for record in records if record.patch_generated]
    target_passed = [record for record in records if record.target_tests_passed]
    total_tasks = sum(record.task_count for record in records)
    total_worker_context = sum(record.worker_context_chars for record in records)
    by_trial: defaultdict[int, list[BenchmarkRunRecord]] = defaultdict(list)
    for record in records:
        by_trial[record.trial_index].append(record)
    trial_success_rates = [
        _ratio(
            sum(item.status is BenchmarkRunStatus.PASSED for item in trial_records),
            len(trial_records),
        )
        for trial_records in by_trial.values()
    ]
    trial_average_tokens = [
        _average([float(item.total_tokens) for item in trial_records])
        for trial_records in by_trial.values()
    ]
    trial_average_durations = [
        _average([float(item.duration_ms) for item in trial_records])
        for trial_records in by_trial.values()
    ]
    hard_constraint_values = [
        record.route_hard_constraints_satisfied
        for record in records
        if record.route_hard_constraints_satisfied is not None
    ]
    initial_execution_success_rate = _ratio(
        sum(record.initial_route_valid for record in records), count
    )
    return StrategyMetrics(
        agent_strategy=key[0],
        model_strategy=key[1],
        evidence_level=key[2],
        run_count=count,
        trial_count=len(by_trial),
        success_rate=_ratio(
            sum(record.status is BenchmarkRunStatus.PASSED for record in records), count
        ),
        success_rate_stddev=_standard_deviation(trial_success_rates),
        localization_recall=_average(
            [
                _ratio(
                    len(
                        set(record.localization_candidates)
                        & set(task_by_id[record.benchmark_id].gold_paths)
                    ),
                    len(task_by_id[record.benchmark_id].gold_paths),
                )
                for record in records
            ]
        ),
        patch_efficiency=_ratio(
            sum(record.patch_applied and record.patch_authorized for record in generated),
            len(generated),
        ),
        regression_rate=_ratio(
            sum(not record.full_regression_passed for record in target_passed),
            len(target_passed),
        ),
        average_tokens=_average([float(record.total_tokens) for record in records]),
        average_tokens_stddev=_standard_deviation(trial_average_tokens),
        average_duration_ms=_average([float(record.duration_ms) for record in records]),
        average_duration_ms_stddev=_standard_deviation(trial_average_durations),
        average_agent_count=_average([float(record.agent_count) for record in records]),
        invalid_task_rate=_ratio(sum(record.invalid_task_count for record in records), total_tasks),
        conflict_rate=_ratio(sum(record.conflict_count > 0 for record in records), count),
        orchestrator_context_ratio=_ratio(
            sum(record.orchestrator_context_chars for record in records),
            total_worker_context,
        ),
        route_hard_constraint_compliance_rate=(
            _ratio(
                sum(hard_constraint_values),
                len(hard_constraint_values),
            )
            if len(hard_constraint_values) == count
            else None
        ),
        initial_execution_success_rate=initial_execution_success_rate,
        initial_route_success_rate=initial_execution_success_rate,
        fallback_rate=_ratio(sum(record.fallback_count > 0 for record in records), count),
        difficulty_match_rate=_ratio(
            sum(
                record.predicted_difficulty is task_by_id[record.benchmark_id].human_difficulty
                for record in records
            ),
            count,
        ),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _standard_deviation(values: list[float]) -> float:
    return round(pstdev(values), 6) if len(values) >= 2 else 0.0
