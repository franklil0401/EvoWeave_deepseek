"""EvoWeave command-line entry point with Chinese human and JSON output."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from evoweave_ds.application.analysis_service import AnalysisService
from evoweave_ds.application.configuration import EvoWeaveConfig, load_config
from evoweave_ds.application.intake_service import IntakeService
from evoweave_ds.application.reporting_service import ReportingService
from evoweave_ds.application.run_state import JsonRunStateStore
from evoweave_ds.application.runtime_layout import RuntimeLayout
from evoweave_ds.application.update_workflow import (
    SingleTaskUpdateWorkflow,
    UpdateWorkflowOutcome,
    ValidationRunnerFactory,
    prepare_task_plan,
)
from evoweave_ds.benchmarking.models import AgentStrategy, EvidenceLevel, ModelStrategy
from evoweave_ds.benchmarking.planning_audit import PlanningAuditRunner, PlanningAuditWriter
from evoweave_ds.benchmarking.reporting import BenchmarkReportWriter, load_run_records
from evoweave_ds.benchmarking.runner import (
    BenchmarkResultStore,
    BenchmarkRunner,
    selected_tasks,
)
from evoweave_ds.benchmarking.suite_loader import (
    load_benchmark_suite,
    validate_benchmark_suite,
)
from evoweave_ds.domain.enums import ModelAvailability
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import RunId
from evoweave_ds.domain.model_routing import ModelProfile
from evoweave_ds.domain.repository_models import RepositoryProfile
from evoweave_ds.domain.run_models import RunManifest
from evoweave_ds.infrastructure.artifacts.local_store import LocalArtifactStore
from evoweave_ds.infrastructure.models.doctor import ModelDoctor
from evoweave_ds.infrastructure.models.openai_compatible import (
    OpenAICompatibleModelGateway,
    default_provider_configs,
)
from evoweave_ds.infrastructure.persistence.graph_repository import SQLiteOrchestrationStore
from evoweave_ds.infrastructure.persistence.sqlite import SQLiteDatabase
from evoweave_ds.interfaces.schemas import CliEnvelope, CliError
from evoweave_ds.orchestration.checkpointing import CheckpointManager
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.repository.profile_cache import calculate_profile_digest
from evoweave_ds.workspaces.command_policy import LocalWorkspaceCommandRunner
from evoweave_ds.workspaces.docker_workspace import (
    DockerSandboxConfig,
    DockerWorkspaceCommandRunner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evoweave_ds",
        description="面向已有 Python 仓库的动态多 Agent 软件更新系统",
    )
    parser.add_argument("--config", type=Path, help="JSON 兼容 YAML 配置文件")
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init", help="初始化仓库本地运行目录")
    _repository_argument(initialize)
    _json_argument(initialize)

    analyze = subparsers.add_parser("analyze", help="读取固定 commit 并生成仓库画像")
    _change_arguments(analyze)

    run = subparsers.add_parser("run", help="创建运行；不带 --execute 时只完成安全预检")
    _change_arguments(run)
    run.add_argument("--execute", action="store_true", help="允许调用模型并执行更新流水线")
    run.add_argument("--provider", help="覆盖默认模型供应商")
    run.add_argument("--model", help="覆盖默认模型 ID")
    run.add_argument(
        "--trusted-host-validation",
        action="store_true",
        help="仅对可信仓库显式允许宿主机验证；默认要求 Docker 沙箱",
    )
    run.add_argument(
        "--approve-high-risk",
        action="store_true",
        help="显式批准已审查的高风险任务范围",
    )

    status = subparsers.add_parser("status", help="查看一个或全部运行状态")
    _repository_argument(status)
    status.add_argument("--run-id")
    _json_argument(status)

    resume = subparsers.add_parser("resume", help="读取可恢复运行状态")
    _repository_argument(resume)
    resume.add_argument("run_id")
    resume.add_argument("--execute", action="store_true", help="继续执行 analyzed 运行")
    resume.add_argument("--provider", help="覆盖默认模型供应商")
    resume.add_argument("--model", help="覆盖默认模型 ID")
    resume.add_argument("--trusted-host-validation", action="store_true")
    resume.add_argument("--approve-high-risk", action="store_true")
    _json_argument(resume)

    export = subparsers.add_parser("export", help="导出中文 Markdown 与机器 JSON 报告")
    _repository_argument(export)
    export.add_argument("run_id")
    export.add_argument("--output", type=Path)
    _json_argument(export)

    models = subparsers.add_parser("models", help="模型配置与可用性诊断")
    model_subparsers = models.add_subparsers(dest="models_command", required=True)
    doctor = model_subparsers.add_parser("doctor", help="检查三个供应商配置")
    doctor.add_argument("--network", action="store_true", help="显式调用只读模型列表接口")
    _json_argument(doctor)

    benchmark = subparsers.add_parser("benchmark", help="固定评测集校验与结果汇总")
    benchmark_subparsers = benchmark.add_subparsers(
        dest="benchmark_command",
        required=True,
    )
    benchmark_validate = benchmark_subparsers.add_parser(
        "validate",
        help="校验任务、fixture commit 和图片摘要",
    )
    benchmark_validate.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmarks/任务集/第二版任务集.json"),
    )
    benchmark_validate.add_argument("--project-root", type=Path, default=Path("."))
    _json_argument(benchmark_validate)
    benchmark_summarize = benchmark_subparsers.add_parser(
        "summarize",
        help="汇总真实或离线回放结果，不填充缺失数据",
    )
    benchmark_summarize.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmarks/任务集/第二版任务集.json"),
    )
    benchmark_summarize.add_argument("--results", type=Path, required=True)
    benchmark_summarize.add_argument("--output", type=Path, required=True)
    _json_argument(benchmark_summarize)
    benchmark_audit = benchmark_subparsers.add_parser(
        "audit",
        help="在 12 个固定仓库任务上比较三种 Agent 规划，不调用模型",
    )
    benchmark_audit.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmarks/任务集/第二版任务集.json"),
    )
    benchmark_audit.add_argument("--project-root", type=Path, default=Path("."))
    benchmark_audit.add_argument("--output", type=Path, required=True)
    _json_argument(benchmark_audit)
    benchmark_run = benchmark_subparsers.add_parser(
        "run",
        help="对固定任务执行真实模型效果实验，并在每项结束后原子保存记录",
    )
    benchmark_run.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmarks/任务集/第二版任务集.json"),
    )
    benchmark_run.add_argument("--project-root", type=Path, default=Path("."))
    benchmark_run.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/结果/真实模型结果.json"),
    )
    benchmark_run.add_argument("--task", action="append", default=[])
    benchmark_run.add_argument(
        "--agent-strategy",
        choices=tuple(item.value for item in AgentStrategy),
        default=AgentStrategy.ADAPTIVE.value,
    )
    benchmark_run.add_argument(
        "--model-strategy",
        choices=tuple(item.value for item in ModelStrategy),
        default=ModelStrategy.ADAPTIVE.value,
    )
    benchmark_run.add_argument(
        "--all-strategies",
        action="store_true",
        help="显式执行 3×3 全部策略；不指定时只运行所选的一组策略",
    )
    benchmark_run.add_argument(
        "--trials",
        type=_benchmark_trial_count,
        default=1,
        help="每个任务与策略组合的重复运行次数，默认 1，最大 100",
    )
    benchmark_run.add_argument(
        "--provider",
        action="append",
        choices=("deepseek",),
        default=[],
        help="限制模型发现供应商，可重复；默认检查三家",
    )
    _json_argument(benchmark_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    as_json = bool(getattr(arguments, "json", False))
    try:
        config = load_config(arguments.config)
        data, human = _dispatch(arguments, config)
    except DomainError as exc:
        _emit_error(exc.code.value, exc.message, as_json)
        return 2
    except (OSError, ValueError) as exc:
        _emit_error("invalid_input", str(exc), as_json)
        return 2
    if as_json:
        print(CliEnvelope(ok=True, data=data).model_dump_json(indent=2))
    else:
        print(human)
    return 0


def _dispatch(
    arguments: argparse.Namespace,
    config: EvoWeaveConfig,
) -> tuple[dict[str, Any], str]:
    if arguments.command == "models":
        return _models_doctor(arguments, config)
    if arguments.command == "benchmark":
        return _benchmark(arguments, config)
    repository = GitInspector(arguments.repository).repository_root
    layout = RuntimeLayout.create(repository, config)
    run_store = JsonRunStateStore(layout.run_state)
    artifact_store = LocalArtifactStore(layout.artifacts)
    if arguments.command == "init":
        init_data = {"repository": str(repository), "runtime_directory": str(layout.root)}
        return init_data, f"EvoWeave 运行目录已初始化：{layout.root}"
    if arguments.command in {"analyze", "run"}:
        manifest, profile = _analyze(arguments, run_store, artifact_store)
        analysis_data: dict[str, Any] = {
            "run_id": str(manifest.run_id),
            "status": manifest.status.value,
            "base_commit": manifest.change_spec.base_commit,
            "profile_artifact_id": str(manifest.repository_profile_artifact_id),
            "files": len(profile.files),
            "python_symbols": len(profile.symbols),
        }
        if arguments.command == "run" and arguments.execute:
            outcome = _execute_update(
                arguments,
                config,
                layout,
                run_store,
                artifact_store,
                manifest,
                profile,
            )
            analysis_data.update(
                {
                    "status": outcome.manifest.status.value,
                    "final_patch_artifact_id": str(outcome.final_patch.ref.artifact_id),
                    "validation_report_artifact_id": str(
                        outcome.validation_report.report_ref.artifact_id
                        if outcome.validation_report.report_ref is not None
                        else ""
                    ),
                    "validation_accepted": outcome.validation_report.accepted,
                    "agent_count": outcome.agent_count,
                }
            )
            return (
                analysis_data,
                f"运行 {manifest.run_id} 已结束：{outcome.manifest.status.value}",
            )
        human = (
            f"运行 {manifest.run_id} 已完成仓库分析："
            f"{len(profile.files)} 个文件，{len(profile.symbols)} 个 Python 符号。"
        )
        return analysis_data, human
    if arguments.command == "status":
        manifests = (
            (run_store.get(RunId(arguments.run_id)),) if arguments.run_id else run_store.list_all()
        )
        status_data: dict[str, Any] = {
            "runs": [
                {
                    "run_id": str(item.run_id),
                    "status": item.status.value,
                    "message": item.message,
                    "base_commit": item.change_spec.base_commit,
                }
                for item in manifests
            ]
        }
        human = (
            "\n".join(f"{item.run_id}  {item.status.value}  {item.message}" for item in manifests)
            or "当前没有运行记录。"
        )
        return status_data, human
    if arguments.command == "resume":
        manifest = run_store.get(RunId(arguments.run_id))
        if arguments.execute:
            profile = _load_profile(manifest, artifact_store)
            outcome = _execute_update(
                arguments,
                config,
                layout,
                run_store,
                artifact_store,
                manifest,
                profile,
            )
            resume_data = outcome.manifest.model_dump(mode="json")
            return resume_data, f"运行已继续并结束：{outcome.manifest.status.value}"
        resume_data = manifest.model_dump(mode="json")
        return resume_data, f"已恢复运行状态：{manifest.run_id} / {manifest.status.value}"
    if arguments.command == "export":
        manifest = run_store.get(RunId(arguments.run_id))
        output = arguments.output or layout.reports
        checkpoint = CheckpointManager(
            SQLiteOrchestrationStore(SQLiteDatabase(layout.orchestration_database))
        ).load(manifest.run_id)
        markdown_path, json_path = ReportingService().export(
            manifest,
            output,
            checkpoint=checkpoint,
        )
        export_data = {"markdown": str(markdown_path), "json": str(json_path)}
        return export_data, f"报告已导出：\n{markdown_path}\n{json_path}"
    raise ValueError(f"未知命令：{arguments.command}")


def _analyze(
    arguments: argparse.Namespace,
    run_store: JsonRunStateStore,
    artifact_store: LocalArtifactStore,
) -> tuple[RunManifest, RepositoryProfile]:
    change_spec = IntakeService().create(
        repository=arguments.repository,
        objective=arguments.request,
        acceptance_criteria=tuple(arguments.acceptance),
        allowed_paths=tuple(arguments.path),
        forbidden_paths=tuple(arguments.forbid),
    )
    return AnalysisService(run_store=run_store, artifact_store=artifact_store).analyze(change_spec)


def _models_doctor(
    arguments: argparse.Namespace,
    config: EvoWeaveConfig,
) -> tuple[dict[str, Any], str]:
    providers = default_provider_configs()
    gateway = OpenAICompatibleModelGateway(providers)
    results = ModelDoctor(providers, gateway).inspect(network=arguments.network)
    data = {"providers": [item.model_dump(mode="json") for item in results]}
    human = "\n".join(
        (
            f"{item.provider}: Key={'已设置' if item.key_present else '未设置'}，"
            f"网络={'成功' if item.reachable else '失败' if item.reachable is False else '未检查'}"
        )
        for item in results
    )
    return data, human


def _benchmark(
    arguments: argparse.Namespace,
    config: EvoWeaveConfig,
) -> tuple[dict[str, Any], str]:
    if arguments.benchmark_command == "validate":
        validation_report = validate_benchmark_suite(
            arguments.project_root,
            arguments.suite,
        )
        data = validation_report.model_dump(mode="json")
        human = (
            f"任务集 {validation_report.suite_id} 校验通过："
            f"{validation_report.task_count} 个任务、"
            f"{validation_report.fixture_count} 个固定仓库、"
            f"{len(validation_report.verified_commits)} 个固定 commit。"
        )
        return data, human
    if arguments.benchmark_command == "summarize":
        suite, suite_digest = load_benchmark_suite(arguments.suite)
        records = load_run_records(arguments.results)
        markdown_path, json_path = BenchmarkReportWriter().write(
            suite=suite,
            suite_sha256=suite_digest,
            records=records,
            output_root=arguments.output,
        )
        data = {
            "suite_id": suite.suite_id,
            "record_count": len(records),
            "markdown": str(markdown_path),
            "json": str(json_path),
        }
        return data, f"评测汇总已生成：\n{markdown_path}\n{json_path}"
    if arguments.benchmark_command == "audit":
        suite, suite_digest = load_benchmark_suite(arguments.suite)
        audit_report = PlanningAuditRunner(arguments.project_root).run(suite, suite_digest)
        markdown_path, json_path = PlanningAuditWriter().write(
            audit_report,
            arguments.output,
        )
        data = {
            "suite_id": suite.suite_id,
            "task_count": audit_report.task_count,
            "record_count": len(audit_report.records),
            "markdown": str(markdown_path),
            "json": str(json_path),
        }
        return data, f"规划审计已生成：\n{markdown_path}\n{json_path}"
    if arguments.benchmark_command == "run":
        validate_benchmark_suite(arguments.project_root, arguments.suite)
        suite, suite_digest = load_benchmark_suite(arguments.suite)
        providers = default_provider_configs()
        selected_provider_names = set(arguments.provider) or {item.provider for item in providers}
        gateway = OpenAICompatibleModelGateway(providers)
        profiles: list[ModelProfile] = []
        provider_failures: dict[str, str] = {}
        for provider in providers:
            if provider.provider not in selected_provider_names:
                continue
            try:
                profiles.extend(
                    item
                    for item in gateway.available_profiles(provider.provider)
                    if item.availability is ModelAvailability.AVAILABLE
                )
            except DomainError as exc:
                provider_failures[provider.provider] = f"{exc.code.value}: {exc.message}"
        if not profiles:
            raise DomainError(
                ErrorCode.MODEL_UNAVAILABLE,
                "所选供应商没有发现任何可用模型",
                details={"providers": provider_failures},
            )
        tasks = selected_tasks(suite, tuple(arguments.task))
        agent_strategies = (
            tuple(AgentStrategy)
            if arguments.all_strategies
            else (AgentStrategy(arguments.agent_strategy),)
        )
        model_strategies = (
            tuple(ModelStrategy)
            if arguments.all_strategies
            else (ModelStrategy(arguments.model_strategy),)
        )
        store = BenchmarkResultStore(arguments.results)
        runner = BenchmarkRunner(
            project_root=arguments.project_root,
            model_gateway=gateway,
            model_profiles=tuple(profiles),
            suite_sha256=suite_digest,
            config=config.model_copy(update={"runtime_directory": ".runtime-benchmark"}),
            hidden_acceptance_source=suite.hidden_acceptance_source,
            hidden_acceptance_sha256=suite.hidden_acceptance_sha256,
            evidence_output_root=Path(arguments.results).resolve().parent / "运行证据",
        )
        stored_records = store.load()
        if any(item.suite_sha256 != suite_digest for item in stored_records):
            raise DomainError(ErrorCode.INVALID_SPEC, "结果文件包含其他任务集版本")
        if any(item.system_commit != runner.system_commit for item in stored_records):
            raise DomainError(ErrorCode.INVALID_SPEC, "结果文件包含其他系统 Git 提交")
        existing = {
            (
                item.system_commit,
                item.benchmark_id,
                item.agent_strategy,
                item.model_strategy,
                item.evidence_level,
                item.trial_index,
            )
            for item in stored_records
        }
        completed = 0
        skipped = 0
        passed = 0
        for agent_strategy in agent_strategies:
            for model_strategy in model_strategies:
                for trial_index in range(1, arguments.trials + 1):
                    for task in tasks:
                        key = (
                            runner.system_commit,
                            task.benchmark_id,
                            agent_strategy,
                            model_strategy,
                            EvidenceLevel.LIVE_MODEL,
                            trial_index,
                        )
                        if key in existing:
                            skipped += 1
                            continue
                        record = runner.run(
                            task=task,
                            agent_strategy=agent_strategy,
                            model_strategy=model_strategy,
                            evidence_level=EvidenceLevel.LIVE_MODEL,
                            trial_index=trial_index,
                        )
                        store.append(record)
                        existing.add(key)
                        completed += 1
                        passed += record.status.value == "passed"
        data = {
            "suite_id": suite.suite_id,
            "completed": completed,
            "passed": passed,
            "skipped_existing": skipped,
            "results": str(Path(arguments.results).resolve()),
            "available_models": [item.key for item in profiles],
            "provider_failures": provider_failures,
            "requested_trials": arguments.trials,
        }
        return (
            data,
            f"真实模型评测本次完成 {completed} 条，通过 {passed} 条，"
            f"跳过已有记录 {skipped} 条。结果：{data['results']}",
        )
    raise ValueError(f"未知 benchmark 命令：{arguments.benchmark_command}")


def _execute_update(
    arguments: argparse.Namespace,
    config: EvoWeaveConfig,
    layout: RuntimeLayout,
    run_store: JsonRunStateStore,
    artifact_store: LocalArtifactStore,
    manifest: RunManifest,
    profile: RepositoryProfile,
) -> UpdateWorkflowOutcome:
    prepare_task_plan(
        config=config,
        run_store=run_store,
        manifest=manifest,
        profile=profile,
        approve_high_risk=arguments.approve_high_risk,
    )
    provider_name = arguments.provider or config.default_provider
    model_id = arguments.model or config.default_model_id
    runner_factory = _validation_runner_factory(arguments, config)
    providers = default_provider_configs()
    gateway = OpenAICompatibleModelGateway(providers)
    discovered_profiles = gateway.available_profiles(provider_name)
    selected = next((item for item in discovered_profiles if item.model_id == model_id), None)
    if selected is None or selected.availability is not ModelAvailability.AVAILABLE:
        raise DomainError(
            ErrorCode.MODEL_UNAVAILABLE,
            f"当前 Key 未发现可用模型 {provider_name}:{model_id}",
        )
    profiles = tuple(
        item.model_copy(update={"stable_priority": 0 if item.model_id == model_id else 100})
        for item in discovered_profiles
        if item.availability is ModelAvailability.AVAILABLE
    )
    return SingleTaskUpdateWorkflow(
        config=config,
        layout=layout,
        run_store=run_store,
        artifact_store=artifact_store,
        model_gateway=gateway,
        model_profiles=profiles,
        validation_runner_factory=runner_factory,
        approve_high_risk=arguments.approve_high_risk,
    ).execute(manifest, profile)


def _validation_runner_factory(
    arguments: argparse.Namespace,
    config: EvoWeaveConfig,
) -> ValidationRunnerFactory:
    if arguments.trusted_host_validation:
        return lambda lease: LocalWorkspaceCommandRunner(
            lease=lease,
            allowed_commands=("python",),
            allow_host_execution=True,
        )
    _assert_docker_ready(config.sandbox_image)
    docker_config = DockerSandboxConfig(image=config.sandbox_image)
    return lambda lease: DockerWorkspaceCommandRunner(
        lease=lease,
        allowed_commands=("python",),
        config=docker_config,
    )


def _load_profile(
    manifest: RunManifest,
    artifact_store: LocalArtifactStore,
) -> RepositoryProfile:
    artifact_id = manifest.repository_profile_artifact_id
    if artifact_id is None:
        raise ValueError("运行没有仓库画像产物")
    profile = RepositoryProfile.model_validate_json(artifact_store.get_bytes(artifact_id))
    if calculate_profile_digest(profile) != profile.profile_digest:
        raise ValueError("仓库画像摘要校验失败")
    return profile


def _assert_docker_ready(image: str) -> None:
    executable = shutil.which("docker")
    if executable is None:
        raise DomainError(ErrorCode.SANDBOX_UNAVAILABLE, "未找到 Docker CLI")
    try:
        completed = subprocess.run(
            (executable, "image", "inspect", image),
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DomainError(ErrorCode.SANDBOX_UNAVAILABLE, "Docker 环境检查失败") from exc
    if completed.returncode != 0:
        raise DomainError(
            ErrorCode.SANDBOX_UNAVAILABLE,
            f"本地不存在锁定沙箱镜像：{image}",
        )


def _repository_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("repository", nargs="?", default=".", help="目标 Git 仓库")


def _benchmark_trial_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--trials 必须是整数") from exc
    if not 1 <= count <= 100:
        raise argparse.ArgumentTypeError("--trials 必须在 1 到 100 之间")
    return count


def _json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")


def _change_arguments(parser: argparse.ArgumentParser) -> None:
    _repository_argument(parser)
    parser.add_argument("--request", required=True, help="软件更新需求")
    parser.add_argument(
        "--acceptance",
        action="append",
        default=["满足用户需求并通过确定性验证"],
        help="验收条件，可重复",
    )
    parser.add_argument("--path", action="append", default=[], help="允许修改路径，可重复")
    parser.add_argument("--forbid", action="append", default=[], help="禁止路径，可重复")
    _json_argument(parser)


def _emit_error(code: str, message: str, as_json: bool) -> None:
    if as_json:
        print(
            CliEnvelope(
                ok=False,
                error=CliError(code=code, message=message),
            ).model_dump_json(indent=2)
        )
    else:
        print(f"错误 [{code}]：{message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
