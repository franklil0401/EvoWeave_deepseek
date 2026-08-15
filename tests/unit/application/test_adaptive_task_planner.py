from collections.abc import Callable
from pathlib import Path

from evoweave_ds.application.adaptive_task_planner import AdaptiveTaskPlanner
from evoweave_ds.application.analysis_service import AnalysisService
from evoweave_ds.application.configuration import EvoWeaveConfig
from evoweave_ds.application.intake_service import IntakeService
from evoweave_ds.application.run_state import JsonRunStateStore
from evoweave_ds.application.runtime_layout import RuntimeLayout
from evoweave_ds.domain.enums import RiskLevel
from evoweave_ds.infrastructure.artifacts.local_store import LocalArtifactStore


def test_planner_consolidates_small_dependent_modules(
    committed_repository: Callable[[str], Path],
) -> None:
    manifest, profile, config = _analyzed(
        committed_repository,
        objective="同时更新价格计算与结账调用",
        allowed_paths=("src/shop/pricing.py", "src/shop/service.py"),
    )

    plan = AdaptiveTaskPlanner(config).plan(manifest, profile)

    assert plan.agent_count == 1
    assert plan.task_specs[0].write_scope == (
        "src/shop/pricing.py",
        "src/shop/service.py",
    )
    assert plan.task_specs[0].depends_on == ()
    assert "合并低收益分组" in plan.rationale


def test_planner_keeps_large_dependent_modules_split(
    committed_repository: Callable[[str], Path],
) -> None:
    manifest, profile, config = _analyzed(
        committed_repository,
        objective="同步更新价格计算与结账调用",
        allowed_paths=("src/shop/pricing.py", "src/shop/service.py"),
    )

    plan = AdaptiveTaskPlanner(config.model_copy(update={"split_directory_lines": 1})).plan(
        manifest, profile
    )

    assert plan.agent_count == 2
    by_scope = {item.write_scope: item for item in plan.task_specs}
    pricing = by_scope[("src/shop/pricing.py",)]
    service = by_scope[("src/shop/service.py",)]
    assert service.depends_on == (pricing.task_id,)
    assert pricing.depends_on == ()
    assert "达到拆分阈值" in plan.rationale


def test_planner_keeps_explicit_independent_changes_parallel(
    committed_repository: Callable[[str], Path],
) -> None:
    manifest, profile, config = _analyzed(
        committed_repository,
        objective="同时独立更新客户模型与价格匹配",
        allowed_paths=("src/shop/models.py", "src/shop/pricing.py"),
    )

    plan = AdaptiveTaskPlanner(config).plan(manifest, profile)

    assert plan.agent_count == 2
    assert all(not item.depends_on for item in plan.task_specs)
    assert "允许无依赖分组并行" in plan.rationale


def test_planner_marks_security_or_payment_change_as_high_risk(
    committed_repository: Callable[[str], Path],
) -> None:
    manifest, profile, config = _analyzed(
        committed_repository,
        objective="修改支付权限和安全校验",
        allowed_paths=("src/shop/pricing.py",),
    )

    plan = AdaptiveTaskPlanner(config).plan(manifest, profile)

    assert plan.task_specs[0].risk_level is RiskLevel.HIGH


def _analyzed(
    committed_repository: Callable[[str], Path],
    *,
    objective: str,
    allowed_paths: tuple[str, ...],
):
    repository = committed_repository("multi_module")
    config = EvoWeaveConfig(runtime_directory=".runtime-planner")
    layout = RuntimeLayout.create(repository, config)
    artifact_store = LocalArtifactStore(layout.artifacts)
    run_store = JsonRunStateStore(layout.run_state)
    change = IntakeService().create(
        repository=repository,
        objective=objective,
        acceptance_criteria=("通过确定性验证",),
        allowed_paths=allowed_paths,
    )
    manifest, profile = AnalysisService(
        run_store=run_store,
        artifact_store=artifact_store,
    ).analyze(change)
    return manifest, profile, config
