import json
from collections.abc import Callable
from pathlib import Path

import pytest

from evoweave_ds.interfaces.cli import build_parser, main


def test_cli_analyze_status_resume_and_export_are_cross_process_durable(
    committed_repository: Callable[[str], Path],
    capsys,
) -> None:
    repository = committed_repository("single_module")

    assert (
        main(
            [
                "analyze",
                str(repository),
                "--request",
                "修改 calculator.py 的折扣逻辑",
                "--path",
                "calculator.py",
                "--json",
            ]
        )
        == 0
    )
    analyzed = json.loads(capsys.readouterr().out)
    run_id = analyzed["data"]["run_id"]
    assert analyzed["data"]["status"] == "analyzed"

    assert main(["status", str(repository), "--run-id", run_id, "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["data"]["runs"][0]["status"] == "analyzed"

    assert main(["resume", str(repository), run_id, "--json"]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["data"]["run_id"] == run_id

    assert main(["export", str(repository), run_id, "--json"]) == 0
    exported = json.loads(capsys.readouterr().out)
    markdown = Path(exported["data"]["markdown"])
    machine_json = Path(exported["data"]["json"])
    assert markdown.name.startswith("运行报告-")
    assert markdown.exists()
    assert machine_json.exists()


def test_cli_models_doctor_default_is_offline_and_does_not_print_key_values(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("Deepseek_api_key", "do-not-print-this-value")

    assert main(["models", "doctor", "--json"]) == 0

    output = capsys.readouterr().out
    data = json.loads(output)
    assert "do-not-print-this-value" not in output
    assert data["data"]["providers"][0]["key_present"] is True
    assert data["data"]["providers"][0]["network_checked"] is False


def test_cli_high_risk_preflight_waits_before_docker_or_model_discovery(
    committed_repository: Callable[[str], Path],
    capsys,
) -> None:
    repository = committed_repository("single_module")

    assert (
        main(
            [
                "run",
                str(repository),
                "--request",
                "修改支付权限安全校验",
                "--path",
                "calculator.py",
                "--execute",
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().out)
    assert error["error"]["code"] == "approval_required"

    assert main(["status", str(repository), "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["data"]["runs"][0]["status"] == "waiting_for_input"


@pytest.mark.skip(reason="阶段 3 重建真实仓库需求集（web_learning_tool）后恢复本测试")
def test_cli_benchmark_validates_fixed_suite_and_keeps_empty_results_pending(
    tmp_path: Path,
    capsys,
) -> None:
    project_root = Path(__file__).parents[2]
    suite = project_root / "benchmarks/任务集/第二版任务集.json"
    results = project_root / "benchmarks/结果/结果模板.json"

    assert (
        main(
            [
                "benchmark",
                "validate",
                "--project-root",
                str(project_root),
                "--suite",
                str(suite),
                "--json",
            ]
        )
        == 0
    )
    validation = json.loads(capsys.readouterr().out)
    assert validation["data"]["task_count"] == 12

    assert (
        main(
            [
                "benchmark",
                "summarize",
                "--suite",
                str(suite),
                "--results",
                str(results),
                "--output",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(Path(summary["data"]["json"]).read_text(encoding="utf-8"))
    assert summary["data"]["record_count"] == 0
    assert report["go_no_go"]["status"] == "pending"


def test_cli_benchmark_run_defaults_to_one_adaptive_strategy_pair() -> None:
    arguments = build_parser().parse_args(["benchmark", "run", "--task", "bench-01-single-file"])

    assert arguments.agent_strategy == "adaptive_agent"
    assert arguments.model_strategy == "adaptive_model"
    assert arguments.all_strategies is False
    assert arguments.trials == 1
    assert arguments.task == ["bench-01-single-file"]


def test_cli_benchmark_run_accepts_bounded_trial_count() -> None:
    arguments = build_parser().parse_args(["benchmark", "run", "--trials", "3"])

    assert arguments.trials == 3

    with pytest.raises(SystemExit):
        build_parser().parse_args(["benchmark", "run", "--trials", "0"])
