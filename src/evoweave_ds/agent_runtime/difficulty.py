"""Explainable rule-based difficulty assessment for stage 1."""

from pydantic import Field

from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.enums import RiskLevel, TaskDifficulty
from evoweave_ds.domain.identifiers import EvidenceId
from evoweave_ds.domain.model_routing import DifficultyAssessment


class TaskSignals(DomainModel):
    affected_files: int = Field(default=0, ge=0)
    affected_symbols: int = Field(default=0, ge=0)
    dependency_depth: int = Field(default=0, ge=0)
    crosses_modules: bool = False
    scope_is_unknown: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    evidence_ids: tuple[EvidenceId, ...] = ()


class RuleBasedDifficultyAssessor:
    def assess(self, signals: TaskSignals) -> DifficultyAssessment:
        score = 0
        reasons: list[str] = []
        if signals.affected_files >= 8:
            score += 4
            reasons.append("影响至少 8 个文件")
        elif signals.affected_files >= 3:
            score += 2
            reasons.append("影响至少 3 个文件")
        if signals.affected_symbols >= 12:
            score += 2
            reasons.append("影响符号较多")
        if signals.dependency_depth >= 4:
            score += 3
            reasons.append("依赖深度至少为 4")
        elif signals.dependency_depth >= 2:
            score += 1
            reasons.append("存在跨层依赖")
        if signals.crosses_modules:
            score += 2
            reasons.append("跨模块修改")
        if signals.scope_is_unknown:
            score += 4
            reasons.append("影响范围未知")
        if signals.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            score += 4
            reasons.append("属于高风险变更")
        elif signals.risk_level is RiskLevel.MEDIUM:
            score += 1
            reasons.append("属于中风险变更")

        if score <= 2:
            difficulty = TaskDifficulty.LOW
        elif score <= 6:
            difficulty = TaskDifficulty.MEDIUM
        else:
            difficulty = TaskDifficulty.HIGH
        rationale = f"规则评分 {score}；" + ("；".join(reasons) if reasons else "无复杂度加分项")
        return DifficultyAssessment(
            difficulty=difficulty,
            rationale=rationale,
            evidence_ids=signals.evidence_ids,
            version=1,
        )
