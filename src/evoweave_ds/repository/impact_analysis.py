"""Turn a user request into evidence-backed impact and difficulty estimates."""

import re
from collections import defaultdict

from evoweave_ds.domain.enums import InputModality, TaskDifficulty
from evoweave_ds.domain.identifiers import EvidenceId, SpecId, TaskId
from evoweave_ds.domain.model_routing import DifficultyAssessment, ModelRequirement
from evoweave_ds.domain.repository_models import (
    ImpactAnalysis,
    ImpactCandidate,
    RepositoryProfile,
    RepositoryTaskAssessment,
    RequirementClues,
)
from evoweave_ds.repository.dependency_graph import dependency_neighbors
from evoweave_ds.repository.evidence_builder import deterministic_evidence_id
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.repository.lexical_search import LexicalSearcher

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}")
_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+"
)
_STOP_WORDS = {
    "and",
    "for",
    "from",
    "need",
    "the",
    "this",
    "with",
    "修改",
    "更新",
    "功能",
    "需要",
    "实现",
    "增加",
    "支持",
}
_RISK_TERMS = {
    "auth": "身份认证",
    "authentication": "身份认证",
    "authorization": "权限控制",
    "concurrency": "并发",
    "database": "数据库",
    "migration": "数据迁移",
    "payment": "支付",
    "security": "安全",
    "权限": "权限控制",
    "并发": "并发",
    "数据库": "数据库",
    "迁移": "数据迁移",
    "支付": "支付",
    "安全": "安全",
}


class RequirementClueExtractor:
    def extract(
        self,
        objective: str,
        acceptance_criteria: tuple[str, ...] = (),
    ) -> RequirementClues:
        text = "\n".join((objective, *acceptance_criteria))
        paths = tuple(dict.fromkeys(match.group(0) for match in _PATH_PATTERN.finditer(text)))
        raw_terms = [match.group(0) for match in _TOKEN_PATTERN.finditer(text)]
        expanded: list[str] = []
        for term in raw_terms:
            expanded.append(term)
            if "_" in term:
                expanded.extend(part for part in term.split("_") if len(part) >= 3)
        terms = tuple(
            dict.fromkeys(
                term
                for term in expanded
                if term.casefold() not in _STOP_WORDS and term not in paths
            )
        )[:64]
        symbols = tuple(
            term
            for term in terms
            if "_" in term or "." in term or (term.isascii() and term[:1].isupper())
        )
        if not terms and not paths:
            terms = (objective[:512],)
        return RequirementClues(terms=terms, symbols=symbols, paths=paths)


class RepositoryImpactAnalyzer:
    def __init__(self) -> None:
        self._clue_extractor = RequirementClueExtractor()
        self._searcher = LexicalSearcher()

    def analyze(
        self,
        *,
        inspector: GitInspector,
        profile: RepositoryProfile,
        objective: str,
        acceptance_criteria: tuple[str, ...] = (),
        max_candidates: int = 20,
    ) -> ImpactAnalysis:
        if profile.base_commit != inspector.base_commit:
            raise ValueError("仓库画像与 GitInspector 必须指向同一 commit")
        clues = self._clue_extractor.extract(objective, acceptance_criteria)
        hits = self._searcher.search(inspector=inspector, files=profile.files, clues=clues)
        scores: dict[str, int] = defaultdict(int)
        reasons: dict[str, set[str]] = defaultdict(set)
        evidence_ids: dict[str, set[EvidenceId]] = defaultdict(set)
        file_evidence = {
            item.path: item.evidence_id
            for item in profile.evidence
            if item.path is not None and item.line_start is None
        }

        for path_clue in clues.paths:
            for file in profile.files:
                if file.path == path_clue or file.path.endswith(f"/{path_clue}"):
                    _add_score(
                        file.path,
                        12,
                        f"需求显式提到路径 {path_clue}",
                        file_evidence.get(file.path),
                        scores,
                        reasons,
                        evidence_ids,
                    )
        for hit in hits:
            _add_score(
                hit.path,
                4,
                f"源码命中词法线索 {hit.term}",
                hit.evidence_id,
                scores,
                reasons,
                evidence_ids,
            )
        folded_terms = {term.casefold() for term in (*clues.terms, *clues.symbols)}
        for symbol in profile.symbols:
            name = symbol.name.casefold()
            qualified_name = symbol.qualified_name.casefold()
            exact = name in folded_terms or qualified_name in folded_terms
            partial = any(term in name or term in qualified_name for term in folded_terms)
            if exact or partial:
                _add_score(
                    symbol.path,
                    9 if exact else 5,
                    f"符号索引命中 {symbol.qualified_name}",
                    symbol.evidence_id,
                    scores,
                    reasons,
                    evidence_ids,
                )

        module_to_path = {
            file.module_name: file.path for file in profile.files if file.module_name is not None
        }
        seed_modules = {
            module
            for module, path in module_to_path.items()
            if path in scores and scores[path] >= 4
        }
        neighbor_depths = dependency_neighbors(seed_modules, profile.dependencies, max_depth=1)
        dependency_evidence: dict[tuple[str, str], EvidenceId] = {}
        for edge in profile.dependencies:
            dependency_evidence[(edge.importer_module, edge.imported_module)] = edge.evidence_id
            dependency_evidence[(edge.imported_module, edge.importer_module)] = edge.evidence_id
        for module, depth in neighbor_depths.items():
            if depth == 0 or module not in module_to_path:
                continue
            source_module = next(
                (seed for seed in sorted(seed_modules) if (seed, module) in dependency_evidence),
                None,
            )
            evidence_id = (
                dependency_evidence.get((source_module, module))
                if source_module is not None
                else None
            )
            _add_score(
                module_to_path[module],
                2,
                "与直接命中模块相邻",
                evidence_id or file_evidence.get(module_to_path[module]),
                scores,
                reasons,
                evidence_ids,
            )

        ordered_paths = sorted(scores, key=lambda path: (-scores[path], path))[:max_candidates]
        candidates = tuple(
            ImpactCandidate(
                path=path,
                score=scores[path],
                reasons=tuple(sorted(reasons[path])),
                evidence_ids=tuple(sorted(evidence_ids[path], key=str)),
            )
            for path in ordered_paths
            if evidence_ids[path]
        )
        confidence, ambiguity = _confidence(candidates)
        candidate_modules = {
            module
            for module, path in module_to_path.items()
            if path in {item.path for item in candidates}
        }
        fan_out = max(
            (
                sum(
                    1
                    for edge in profile.dependencies
                    if edge.importer_module == module and edge.imported_module in candidate_modules
                )
                for module in candidate_modules
            ),
            default=0,
        )
        risk_signals = _risk_signals("\n".join((objective, *acceptance_criteria)))
        return ImpactAnalysis(
            base_commit=profile.base_commit,
            clues=clues,
            search_hits=hits,
            candidates=candidates,
            confidence=confidence,
            ambiguity=ambiguity,
            dependency_fan_out=fan_out,
            risk_signals=risk_signals,
        )


class RepositoryDifficultyAssessor:
    def assess(
        self,
        *,
        task_id: TaskId,
        impact: ImpactAnalysis,
        required_modalities: tuple[InputModality, ...] = (InputModality.TEXT,),
        previous_requirement: ModelRequirement | None = None,
    ) -> RepositoryTaskAssessment:
        score = 0
        reasons: list[str] = []
        explicit_candidates = {
            candidate.path
            for candidate in impact.candidates
            if any(reason.startswith("需求显式提到路径") for reason in candidate.reasons)
        }
        impacted_files = len(explicit_candidates) if explicit_candidates else len(impact.candidates)
        if not impact.candidates:
            score += 7
            reasons.append("没有可定位的候选影响范围")
        if impacted_files >= 6:
            score += 5
            reasons.append("候选影响文件至少 6 个")
        elif impacted_files >= 2:
            score += 2
            reasons.append("候选影响文件跨越多个文件")
        if impacted_files > 1 and impact.dependency_fan_out >= 4:
            score += 3
            reasons.append("候选模块依赖扇出至少为 4")
        elif impacted_files > 1 and impact.dependency_fan_out >= 2:
            score += 1
            reasons.append("候选模块存在依赖扇出")
        if impact.ambiguity >= 0.6 or impact.confidence < 0.4:
            score += 4
            reasons.append("定位歧义高或置信度低")
        elif impact.ambiguity >= 0.35:
            score += 1
            reasons.append("定位存在一定歧义")
        if impact.risk_signals:
            score += 4
            reasons.append("包含风险信号：" + "、".join(impact.risk_signals))

        if score <= 2:
            difficulty = TaskDifficulty.LOW
        elif score <= 6:
            difficulty = TaskDifficulty.MEDIUM
        else:
            difficulty = TaskDifficulty.HIGH
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for candidate in impact.candidates
                for evidence_id in candidate.evidence_ids
            )
        )
        requirement = _version_model_requirement(
            task_id=task_id,
            difficulty=difficulty,
            required_modalities=required_modalities,
            previous=previous_requirement,
        )
        difficulty_assessment = DifficultyAssessment(
            difficulty=difficulty,
            rationale=f"仓库证据规则评分 {score}；"
            + ("；".join(reasons) if reasons else "单一、明确且低风险的候选范围"),
            evidence_ids=evidence_ids,
            version=requirement.version,
        )
        return RepositoryTaskAssessment(
            difficulty=difficulty_assessment,
            impact=impact,
            model_requirement=requirement,
        )


def _version_model_requirement(
    *,
    task_id: TaskId,
    difficulty: TaskDifficulty,
    required_modalities: tuple[InputModality, ...],
    previous: ModelRequirement | None,
) -> ModelRequirement:
    context_tokens = {
        TaskDifficulty.LOW: 8_000,
        TaskDifficulty.MEDIUM: 32_000,
        TaskDifficulty.HIGH: 64_000,
    }[difficulty]
    output_tokens = {
        TaskDifficulty.LOW: 2_000,
        TaskDifficulty.MEDIUM: 4_000,
        TaskDifficulty.HIGH: 8_000,
    }[difficulty]
    if previous is not None:
        if previous.task_id != task_id:
            raise ValueError("previous_requirement 必须属于同一 task_id")
        changed = (
            previous.difficulty is not difficulty
            or previous.required_modalities != required_modalities
            or previous.min_context_tokens != context_tokens
            or previous.min_output_tokens != output_tokens
            or not previous.requires_tool_calling
            or not previous.requires_structured_output
            or previous.requires_thinking != (difficulty is TaskDifficulty.HIGH)
        )
        if not changed:
            return previous
        return ModelRequirement.model_validate(
            {
                **previous.model_dump(),
                "difficulty": difficulty,
                "required_modalities": required_modalities,
                "min_context_tokens": context_tokens,
                "min_output_tokens": output_tokens,
                "requires_tool_calling": True,
                "requires_structured_output": True,
                "requires_thinking": difficulty is TaskDifficulty.HIGH,
                "version": previous.version + 1,
            }
        )
    requirement_id = SpecId(
        f"spec_{deterministic_evidence_id(task_id, 'model-requirement').split('_', 1)[1]}"
    )
    return ModelRequirement(
        requirement_id=requirement_id,
        task_id=task_id,
        difficulty=difficulty,
        required_modalities=required_modalities,
        min_context_tokens=context_tokens,
        min_output_tokens=output_tokens,
        requires_tool_calling=True,
        requires_structured_output=True,
        requires_thinking=difficulty is TaskDifficulty.HIGH,
        version=1,
    )


def _add_score(
    path: str,
    score: int,
    reason: str,
    evidence_id: EvidenceId | None,
    scores: dict[str, int],
    reasons: dict[str, set[str]],
    evidence_ids: dict[str, set[EvidenceId]],
) -> None:
    scores[path] += score
    reasons[path].add(reason)
    if evidence_id is not None:
        evidence_ids[path].add(evidence_id)


def _confidence(candidates: tuple[ImpactCandidate, ...]) -> tuple[float, float]:
    if not candidates:
        return 0.0, 1.0
    top = candidates[0].score
    second = candidates[1].score if len(candidates) > 1 else 0
    gap = max(0, top - second)
    confidence = min(0.95, 0.4 + min(top, 12) / 30 + min(gap, 8) / 40)
    tie_count = sum(1 for item in candidates if item.score == top)
    ambiguity = min(1.0, max(0.0, 1.0 - confidence + (tie_count - 1) * 0.15))
    return round(confidence, 3), round(ambiguity, 3)


def _risk_signals(text: str) -> tuple[str, ...]:
    folded = text.casefold()
    return tuple(sorted({label for term, label in _RISK_TERMS.items() if term in folded}))
