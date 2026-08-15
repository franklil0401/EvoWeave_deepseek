import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from threading import Barrier

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
from evoweave_ds.domain.ports import ModelRequest, ModelResponse
from evoweave_ds.infrastructure.artifacts.local_store import LocalArtifactStore
from evoweave_ds.infrastructure.models.fake import ScriptedModelGateway
from evoweave_ds.infrastructure.persistence.graph_repository import SQLiteOrchestrationStore
from evoweave_ds.infrastructure.persistence.sqlite import SQLiteDatabase
from evoweave_ds.orchestration.checkpointing import CheckpointManager
from evoweave_ds.workspaces.command_policy import LocalWorkspaceCommandRunner

_SCOPE_PATTERN = re.compile(r"本任务只负责写范围：([^；]+)；")


class ScopeAwareModelGateway:
    def __init__(
        self,
        profile: ModelProfile,
        *,
        first_call_barrier: Barrier | None = None,
    ) -> None:
        self._profile = profile
        self._first_call_barrier = first_call_barrier
        self._steps: defaultdict[str, int] = defaultdict(int)
        self.requests: list[ModelRequest] = []

    def list_profiles(self) -> tuple[ModelProfile, ...]:
        return (self._profile,)

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        match = _SCOPE_PATTERN.search(request.messages[1])
        assert match is not None
        path = match.group(1)
        step = self._steps[path]
        self._steps[path] += 1
        if step == 0:
            if self._first_call_barrier is not None:
                self._first_call_barrier.wait(timeout=5)
            payload: dict[str, object] = {
                "action": "tool",
                "tool_name": "file.read",
                "arguments": {"path": path},
            }
        elif step == 1:
            payload = {
                "action": "tool",
                "tool_name": "file.write",
                "arguments": {"path": path, "content": _updated_content(path)},
            }
        else:
            payload = {
                "action": "finish",
                "status": "succeeded",
                "summary": f"已更新 {path}",
            }
        return ModelResponse(
            model_key=request.model_key,
            text=json.dumps(payload, ensure_ascii=False),
            input_tokens=10,
            output_tokens=5,
        )


def test_two_independent_scopes_create_two_agents_and_integrate_both_patches(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("single_module")
    (repository / "formatter.py").write_text(
        "def format_label(value: str) -> str:\n    return value.strip()\n",
        encoding="utf-8",
    )
    (repository / "tests" / "test_formatter.py").write_text(
        "from formatter import format_label\n\n\n"
        "def test_format_label() -> None:\n"
        '    assert format_label(" vip ") == "vip"\n',
        encoding="utf-8",
    )
    _git(repository, "add", "formatter.py", "tests/test_formatter.py")
    _git(repository, "commit", "-m", "add formatter fixture")
    config = EvoWeaveConfig(runtime_directory=".runtime-multi")
    layout = RuntimeLayout.create(repository, config)
    artifact_store = LocalArtifactStore(layout.artifacts)
    run_store = JsonRunStateStore(layout.run_state)
    change = IntakeService().create(
        repository=repository,
        objective="两个改动相互独立且可并行：折扣客户类型忽略大小写，并统一标签为小写",
        acceptance_criteria=("全部测试通过",),
        allowed_paths=("calculator.py", "formatter.py"),
    )
    manifest, repository_profile = AnalysisService(
        run_store=run_store,
        artifact_store=artifact_store,
    ).analyze(change)
    profile = _profile()
    gateway = ScopeAwareModelGateway(profile, first_call_barrier=Barrier(2))

    outcome = SingleTaskUpdateWorkflow(
        config=config,
        layout=layout,
        run_store=run_store,
        artifact_store=artifact_store,
        model_gateway=gateway,
        model_profiles=(profile,),
        validation_runner_factory=lambda lease: LocalWorkspaceCommandRunner(
            lease=lease,
            allowed_commands=("python",),
            allow_host_execution=True,
        ),
    ).execute(manifest, repository_profile)

    assert outcome.manifest.status is RunStatus.COMPLETED
    assert outcome.agent_count == 2
    assert outcome.final_patch.changed_paths == ("calculator.py", "formatter.py")
    final_diff = artifact_store.get_bytes(outcome.final_patch.ref.artifact_id).decode()
    assert "customer_type.upper()" in final_diff
    assert ".lower()" in final_diff
    checkpoint = CheckpointManager(
        SQLiteOrchestrationStore(SQLiteDatabase(layout.orchestration_database))
    ).load(manifest.run_id)
    assert checkpoint is not None
    assert len(checkpoint.execution_specs) == 2
    assert len(checkpoint.allocation_decisions[0].selected_task_ids) == 2


def test_high_risk_change_waits_for_explicit_approval_without_model_call(
    committed_repository: Callable[[str], Path],
) -> None:
    repository = committed_repository("single_module")
    config = EvoWeaveConfig(runtime_directory=".runtime-approval")
    layout = RuntimeLayout.create(repository, config)
    artifact_store = LocalArtifactStore(layout.artifacts)
    run_store = JsonRunStateStore(layout.run_state)
    change = IntakeService().create(
        repository=repository,
        objective="修改支付权限安全校验",
        acceptance_criteria=("安全测试通过",),
        allowed_paths=("calculator.py",),
    )
    manifest, repository_profile = AnalysisService(
        run_store=run_store,
        artifact_store=artifact_store,
    ).analyze(change)
    profile = _profile()
    gateway = ScriptedModelGateway(profiles=(profile,))

    with pytest.raises(DomainError) as captured:
        SingleTaskUpdateWorkflow(
            config=config,
            layout=layout,
            run_store=run_store,
            artifact_store=artifact_store,
            model_gateway=gateway,
            model_profiles=(profile,),
            validation_runner_factory=lambda _lease: pytest.fail("等待批准时不应创建验证执行器"),
        ).execute(manifest, repository_profile)

    assert captured.value.code is ErrorCode.APPROVAL_REQUIRED
    waiting = run_store.get(manifest.run_id)
    assert waiting.status is RunStatus.WAITING_FOR_INPUT
    assert waiting.error_code is None
    assert "等待人工批准" in waiting.message
    assert gateway.requests == []


def _profile() -> ModelProfile:
    from datetime import UTC, datetime

    return ModelProfile(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        tier=ModelTier.HIGH,
        availability=ModelAvailability.AVAILABLE,
        input_modalities=(InputModality.TEXT,),
        context_window_tokens=128_000,
        max_output_tokens=8_000,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_thinking=True,
        checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _updated_content(path: str) -> str:
    if path == "calculator.py":
        return (
            "def calculate_discount(total: float, customer_type: str) -> float:\n"
            '    if customer_type.upper() == "VIP":\n'
            "        return total * 0.9\n"
            "    return total\n"
        )
    if path == "formatter.py":
        return "def format_label(value: str) -> str:\n    return value.strip().lower()\n"
    raise AssertionError(f"unexpected path: {path}")


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ("git", "-C", str(repository), *args),
        check=True,
        capture_output=True,
        text=True,
    )
