"""Evidence-backed task count, modality, and dependency planning without fixed roles."""

from dataclasses import dataclass
from typing import Protocol

from evoweave_ds.application.configuration import EvoWeaveConfig
from evoweave_ds.domain.enums import InputModality, RiskLevel, TaskDifficulty
from evoweave_ds.domain.identifiers import SpecId, TaskId
from evoweave_ds.domain.repository_models import (
    RepositoryFile,
    RepositoryProfile,
    RepositoryTaskAssessment,
    difficulty_rank,
)
from evoweave_ds.domain.run_models import RunManifest
from evoweave_ds.domain.task_spec import TaskSpec
from evoweave_ds.domain.validation import path_is_within_scopes
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.repository.impact_analysis import (
    RepositoryDifficultyAssessor,
    RepositoryImpactAnalyzer,
)

_PARALLEL_OBJECTIVE_TERMS = ("并行", "同时", "独立", "分别", "互不依赖", "可并发")
_HIGH_COMPLEXITY_TERMS = (
    "架构",
    "数据流",
    "冲突",
    "迁移",
    "并发",
    "权限",
    "安全",
    "支付",
    "已有失败",
)


@dataclass(frozen=True, slots=True)
class AdaptiveTaskPlan:
    task_specs: tuple[TaskSpec, ...]
    rationale: str

    @property
    def agent_count(self) -> int:
        return len(self.task_specs)


class TaskPlanner(Protocol):
    def plan(
        self,
        manifest: RunManifest,
        profile: RepositoryProfile,
    ) -> AdaptiveTaskPlan: ...


class AdaptiveTaskPlanner:
    def __init__(self, config: EvoWeaveConfig) -> None:
        self._config = config

    def plan(
        self,
        manifest: RunManifest,
        profile: RepositoryProfile,
    ) -> AdaptiveTaskPlan:
        change = manifest.change_spec
        initial_groups = _write_groups(
            change.allowed_paths,
            profile.files,
            max_tasks=self._config.max_dynamic_tasks,
            split_directory_lines=self._config.split_directory_lines,
        )
        groups, topology_reason = _select_execution_groups(
            initial_groups,
            objective=change.objective,
            files=profile.files,
            profile=profile,
            split_directory_lines=self._config.split_directory_lines,
        )
        task_ids = tuple(TaskId.new() for _ in groups)
        difficulty_floor, floor_reasons = _task_structure_difficulty_floor(
            objective=change.objective,
            acceptance_criteria=change.acceptance_criteria,
            allowed_paths=change.allowed_paths,
            groups=groups,
            files=profile.files,
        )
        readable_paths = tuple(item.path for item in profile.files if item.line_count > 0)
        context_artifacts = (
            (manifest.repository_profile_artifact_id,)
            if manifest.repository_profile_artifact_id is not None
            else ()
        )
        inspector = GitInspector(change.repository, change.base_commit)
        specs: list[TaskSpec] = []
        for _index, (task_id, write_scope) in enumerate(zip(task_ids, groups, strict=True)):
            goal = (
                f"{change.objective}\n"
                f"本任务只负责写范围：{', '.join(write_scope)}；"
                "可以读取授权仓库证据，但不得修改其他路径。"
            )
            modalities = (InputModality.TEXT,)
            impact = RepositoryImpactAnalyzer().analyze(
                inspector=inspector,
                profile=profile,
                objective=goal,
                acceptance_criteria=change.acceptance_criteria,
            )
            assessment = RepositoryDifficultyAssessor().assess(
                task_id=task_id,
                impact=impact,
                required_modalities=modalities,
            )
            assessment = _apply_difficulty_floor(
                assessment,
                minimum=difficulty_floor,
                reasons=floor_reasons,
            )
            read_scope = tuple(dict.fromkeys((*readable_paths, *write_scope)))
            specs.append(
                TaskSpec(
                    spec_id=SpecId.new(),
                    task_id=task_id,
                    change_spec_id=change.spec_id,
                    goal=goal,
                    base_commit=change.base_commit,
                    acceptance_criteria=change.acceptance_criteria,
                    context_artifact_ids=context_artifacts,
                    read_scope=read_scope,
                    write_scope=write_scope,
                    required_modalities=modalities,
                    difficulty=assessment.difficulty,
                    model_requirement=assessment.model_requirement,
                    risk_level=_risk_level(
                        assessment.difficulty.difficulty,
                        assessment.impact.risk_signals,
                    ),
                )
            )

        dependencies = _task_dependencies(groups, task_ids, profile)
        planned = tuple(
            spec.model_copy(update={"depends_on": dependencies[index]})
            for index, spec in enumerate(specs)
        )
        relation_count = sum(len(item.depends_on) for item in planned)
        rationale = (
            f"按 {len(change.allowed_paths)} 个用户写范围、固定 commit 文件规模和模块依赖，"
            f"生成 {len(planned)} 个临时任务、{relation_count} 条依赖；{topology_reason}。"
        )
        return AdaptiveTaskPlan(task_specs=planned, rationale=rationale)


def _task_structure_difficulty_floor(
    *,
    objective: str,
    acceptance_criteria: tuple[str, ...],
    allowed_paths: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
    files: tuple[RepositoryFile, ...],
) -> tuple[TaskDifficulty, tuple[str, ...]]:
    reasons: list[str] = []
    minimum = TaskDifficulty.LOW
    if len(groups) >= 2:
        minimum = TaskDifficulty.MEDIUM
        reasons.append("任务需要协调多个独立写范围")
    broad_scopes = tuple(scope for scope in allowed_paths if _is_broad_scope(scope, files))
    if broad_scopes:
        minimum = TaskDifficulty.HIGH
        reasons.append("用户授权的是目录或宽泛范围：" + "、".join(broad_scopes))
    text = "\n".join((objective, *acceptance_criteria)).casefold()
    matched_terms = tuple(term for term in _HIGH_COMPLEXITY_TERMS if term in text)
    if matched_terms:
        minimum = TaskDifficulty.HIGH
        reasons.append("需求包含高复杂度语义：" + "、".join(matched_terms))
    return minimum, tuple(reasons)


def _is_broad_scope(scope: str, files: tuple[RepositoryFile, ...]) -> bool:
    if any(item.path == scope for item in files):
        return False
    descendants = tuple(item for item in files if path_is_within_scopes(item.path, (scope,)))
    leaf = scope.rsplit("/", 1)[-1]
    return len(descendants) >= 2 or "." not in leaf


def _apply_difficulty_floor(
    assessment: RepositoryTaskAssessment,
    *,
    minimum: TaskDifficulty,
    reasons: tuple[str, ...],
) -> RepositoryTaskAssessment:
    current = assessment.difficulty.difficulty
    if difficulty_rank(current) >= difficulty_rank(minimum):
        return assessment
    context_tokens = {
        TaskDifficulty.LOW: 8_000,
        TaskDifficulty.MEDIUM: 32_000,
        TaskDifficulty.HIGH: 64_000,
    }[minimum]
    output_tokens = {
        TaskDifficulty.LOW: 2_000,
        TaskDifficulty.MEDIUM: 4_000,
        TaskDifficulty.HIGH: 8_000,
    }[minimum]
    rationale = assessment.difficulty.rationale + "；结构性难度下限：" + "；".join(reasons)
    difficulty = assessment.difficulty.model_copy(
        update={"difficulty": minimum, "rationale": rationale}
    )
    requirement = assessment.model_requirement.model_copy(
        update={
            "difficulty": minimum,
            "min_context_tokens": context_tokens,
            "min_output_tokens": output_tokens,
            "requires_thinking": minimum is TaskDifficulty.HIGH,
        }
    )
    return assessment.model_copy(
        update={"difficulty": difficulty, "model_requirement": requirement}
    )


def _write_groups(
    allowed_paths: tuple[str, ...],
    files: tuple[RepositoryFile, ...],
    *,
    max_tasks: int,
    split_directory_lines: int,
) -> tuple[tuple[str, ...], ...]:
    collapsed = tuple(
        scope
        for scope in allowed_paths
        if not any(
            scope != other and path_is_within_scopes(scope, (other,)) for other in allowed_paths
        )
    )
    expanded: list[str] = []
    for scope in collapsed:
        exact = next((item for item in files if item.path == scope), None)
        candidates = tuple(
            item
            for item in files
            if item.line_count > 0 and path_is_within_scopes(item.path, (scope,))
        )
        if (
            exact is None
            and len(candidates) >= 2
            and sum(item.line_count for item in candidates) >= split_directory_lines
        ):
            expanded.extend(item.path for item in candidates)
        else:
            expanded.append(scope)
    scopes = tuple(dict.fromkeys(expanded))
    if len(scopes) <= max_tasks:
        return tuple((scope,) for scope in scopes)

    line_counts = {item.path: item.line_count for item in files}
    buckets: list[list[str]] = [[] for _ in range(max_tasks)]
    weights = [0] * max_tasks
    for scope in sorted(scopes, key=lambda item: (-line_counts.get(item, 1), item)):
        target = min(range(max_tasks), key=lambda index: (weights[index], index))
        buckets[target].append(scope)
        weights[target] += line_counts.get(scope, 1)
    return tuple(tuple(sorted(bucket)) for bucket in buckets if bucket)


def _select_execution_groups(
    groups: tuple[tuple[str, ...], ...],
    *,
    objective: str,
    files: tuple[RepositoryFile, ...],
    profile: RepositoryProfile,
    split_directory_lines: int,
) -> tuple[tuple[tuple[str, ...], ...], str]:
    """Keep a split only when its independence or scale can repay extra calls."""

    if len(groups) <= 1:
        return groups, "单一写分组无需扩展任务图"

    dependencies = _group_dependency_edges(groups, profile)
    folded_objective = objective.casefold()
    parallel_signal = any(term in folded_objective for term in _PARALLEL_OBJECTIVE_TERMS)
    total_lines = _covered_line_count(groups, files)

    reasons: list[str] = []
    if parallel_signal and not dependencies:
        reasons.append("需求明确允许无依赖分组并行")
    if total_lines >= split_directory_lines:
        reasons.append(f"授权代码量 {total_lines} 行达到拆分阈值 {split_directory_lines} 行")
    if reasons:
        return groups, "保留拆分：" + "；".join(reasons)

    merged = (tuple(scope for group in groups for scope in group),)
    dependency_text = "存在跨分组依赖" if dependencies else "缺少明确独立性证据"
    return (
        merged,
        f"合并低收益分组：{dependency_text}，且授权代码量 {total_lines} 行低于"
        f" {split_directory_lines} 行阈值",
    )


def _covered_line_count(
    groups: tuple[tuple[str, ...], ...],
    files: tuple[RepositoryFile, ...],
) -> int:
    scopes = tuple(scope for group in groups for scope in group)
    covered = {
        item.path: item.line_count
        for item in files
        if item.line_count > 0 and path_is_within_scopes(item.path, scopes)
    }
    return sum(covered.values())


def _risk_level(
    difficulty: TaskDifficulty,
    risk_signals: tuple[str, ...],
) -> RiskLevel:
    if risk_signals:
        return RiskLevel.HIGH
    if difficulty is TaskDifficulty.HIGH:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _task_dependencies(
    groups: tuple[tuple[str, ...], ...],
    task_ids: tuple[TaskId, ...],
    profile: RepositoryProfile,
) -> tuple[tuple[TaskId, ...], ...]:
    candidates = _group_dependency_edges(groups, profile)
    accepted: set[tuple[int, int]] = set()
    for source, target in sorted(candidates, key=lambda item: (groups[item[0]], groups[item[1]])):
        if not _would_create_cycle(accepted, source, target):
            accepted.add((source, target))
    return tuple(
        tuple(task_ids[source] for source, target in sorted(accepted) if target == index)
        for index in range(len(groups))
    )


def _group_dependency_edges(
    groups: tuple[tuple[str, ...], ...],
    profile: RepositoryProfile,
) -> set[tuple[int, int]]:
    module_to_group: dict[str, int] = {}
    for file in profile.files:
        if file.module_name is None:
            continue
        group_index = next(
            (
                index
                for index, scopes in enumerate(groups)
                if path_is_within_scopes(file.path, scopes)
            ),
            None,
        )
        if group_index is not None:
            module_to_group[file.module_name] = group_index

    return {
        (module_to_group[edge.imported_module], module_to_group[edge.importer_module])
        for edge in profile.dependencies
        if edge.imported_module in module_to_group
        and edge.importer_module in module_to_group
        and module_to_group[edge.imported_module] != module_to_group[edge.importer_module]
    }


def _would_create_cycle(
    edges: set[tuple[int, int]],
    source: int,
    target: int,
) -> bool:
    frontier = [target]
    visited: set[int] = set()
    while frontier:
        current = frontier.pop()
        if current == source:
            return True
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(edge_target for edge_source, edge_target in edges if edge_source == current)
    return False
