"""Single-model strategy: model failure is a structured failure with no fallback."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evoweave_ds.application.analysis_service import AnalysisService
from evoweave_ds.application.configuration import EvoWeaveConfig
from evoweave_ds.application.intake_service import IntakeService
from evoweave_ds.application.run_state import JsonRunStateStore
from evoweave_ds.application.runtime_layout import RuntimeLayout
from evoweave_ds.application.update_workflow import SingleTaskUpdateWorkflow
from evoweave_ds.domain.enums import (
    InputModality,
    ModelAvailability,
    ModelTier,
    RunStatus,
)
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.model_routing import ModelProfile
from evoweave_ds.domain.ports import CommandResult, ModelRequest, ModelResponse
from evoweave_ds.infrastructure.artifacts.local_store import LocalArtifactStore
from evoweave_ds.infrastructure.persistence.graph_repository import SQLiteOrchestrationStore
from evoweave_ds.infrastructure.persistence.sqlite import SQLiteDatabase
from evoweave_ds.orchestration.checkpointing import CheckpointManager


class AlwaysFailGateway:
    def __init__(self, profiles: tuple[ModelProfile, ...]) -> None:
        self._profiles = profiles
        self.requests: list[ModelRequest] = []

    def list_profiles(self) -> tuple[ModelProfile, ...]:
        return self._profiles

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        raise DomainError(
            ErrorCode.MODEL_UNAVAILABLE,
            "模拟模型服务不可用",
            details={"model_key": request.model_key},
        )


class AlwaysPassRunner:
    def run(self, argv: tuple[str, ...], *, timeout_seconds: int) -> CommandResult:
        assert timeout_seconds > 0
        return CommandResult(argv=argv, exit_code=0, stdout="passed", duration_ms=1)


def test_model_failure_fails_run_without_creating_new_agent(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("single_module")
    config = EvoWeaveConfig(runtime_directory=".runtime-fallback")
    layout = RuntimeLayout.create(repository, config)
    artifact_store = LocalArtifactStore(layout.artifacts)
    run_store = JsonRunStateStore(layout.run_state)
    change = IntakeService().create(
        repository=repository,
        objective="修改 calculator.py，让客户类型匹配不区分大小写",
        acceptance_criteria=("VIP 折扣保持正确",),
        allowed_paths=("calculator.py",),
    )
    manifest, repository_profile = AnalysisService(
        run_store=run_store,
        artifact_store=artifact_store,
    ).analyze(change)
    flash = _flash_profile()
    gateway = AlwaysFailGateway((flash,))

    with pytest.raises(DomainError) as error:
        SingleTaskUpdateWorkflow(
            config=config,
            layout=layout,
            run_store=run_store,
            artifact_store=artifact_store,
            model_gateway=gateway,
            model_profiles=(flash,),
            validation_runner_factory=lambda _lease: AlwaysPassRunner(),
        ).execute(manifest, repository_profile)

    assert error.value.code is ErrorCode.MODEL_UNAVAILABLE
    checkpoint = CheckpointManager(
        SQLiteOrchestrationStore(SQLiteDatabase(layout.orchestration_database))
    ).load(manifest.run_id)
    assert checkpoint is not None
    assert len(checkpoint.execution_specs) == 1
    assert checkpoint.execution_specs[0].model_routing.selected_model_key == (
        "deepseek:deepseek-v4-flash"
    )
    failed_manifest = run_store.get(manifest.run_id)
    assert failed_manifest.status is RunStatus.FAILED
    assert failed_manifest.error_code is ErrorCode.MODEL_UNAVAILABLE


def test_model_failure_records_single_call_without_retry(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("single_module")
    config = EvoWeaveConfig(runtime_directory=".runtime-fallback-limit")
    layout = RuntimeLayout.create(repository, config)
    artifact_store = LocalArtifactStore(layout.artifacts)
    run_store = JsonRunStateStore(layout.run_state)
    change = IntakeService().create(
        repository=repository,
        objective="修改 calculator.py，让客户类型匹配不区分大小写",
        acceptance_criteria=("VIP 折扣保持正确",),
        allowed_paths=("calculator.py",),
    )
    manifest, repository_profile = AnalysisService(
        run_store=run_store,
        artifact_store=artifact_store,
    ).analyze(change)
    flash = _flash_profile()
    gateway = AlwaysFailGateway((flash,))

    with pytest.raises(DomainError) as error:
        SingleTaskUpdateWorkflow(
            config=config,
            layout=layout,
            run_store=run_store,
            artifact_store=artifact_store,
            model_gateway=gateway,
            model_profiles=(flash,),
            validation_runner_factory=lambda _lease: AlwaysPassRunner(),
        ).execute(manifest, repository_profile)

    assert error.value.code is ErrorCode.MODEL_UNAVAILABLE
    assert len(gateway.requests) == 1
    assert gateway.requests[0].model_key == "deepseek:deepseek-v4-flash"


def _flash_profile() -> ModelProfile:
    return ModelProfile(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        tier=ModelTier.LOW,
        availability=ModelAvailability.AVAILABLE,
        input_modalities=(InputModality.TEXT,),
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_thinking=True,
        stable_priority=0,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
