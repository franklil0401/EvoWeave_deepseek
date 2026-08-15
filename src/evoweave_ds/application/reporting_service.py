"""Export Chinese summaries with the persisted orchestration control record."""

import json
import os
from pathlib import Path

from evoweave_ds.domain.run_models import RunManifest
from evoweave_ds.orchestration.checkpointing import OrchestrationCheckpoint


class ReportingService:
    def export(
        self,
        manifest: RunManifest,
        output_root: Path | str,
        *,
        checkpoint: OrchestrationCheckpoint | None = None,
    ) -> tuple[Path, Path]:
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        markdown_path = root / f"运行报告-{manifest.run_id}.md"
        json_path = root / f"运行报告-{manifest.run_id}.json"
        patch = str(manifest.final_patch_artifact_id or "无")
        validation = str(manifest.validation_report_artifact_id or "无")
        profile = str(manifest.repository_profile_artifact_id or "无")
        agent_count = len(checkpoint.execution_specs) if checkpoint is not None else 0
        task_count = len(checkpoint.task_specs) if checkpoint is not None else 0
        model_routes = (
            "、".join(
                f"{item.agent_id} → {item.model_routing.selected_model_key}"
                for item in checkpoint.execution_specs
            )
            if checkpoint is not None
            else "无"
        )
        markdown = (
            f"# EvoWeave 运行报告\n\n"
            f"- 运行 ID：`{manifest.run_id}`\n"
            f"- 状态：`{manifest.status.value}`\n"
            f"- 仓库：`{manifest.change_spec.repository}`\n"
            f"- 基线：`{manifest.change_spec.base_commit}`\n"
            f"- 目标：{manifest.change_spec.objective}\n"
            f"- 当前信息：{manifest.message}\n"
            f"- 动态任务数：{task_count}\n"
            f"- 实际 Agent 实例数：{agent_count}\n"
            f"- 模型路由：{model_routes}\n"
            f"- 仓库画像产物：`{profile}`\n"
            f"- 最终补丁产物：`{patch}`\n"
            f"- 验证报告产物：`{validation}`\n"
        )
        _atomic_text(markdown_path, markdown)
        machine_payload = manifest.model_dump(mode="json")
        machine_payload["orchestration"] = (
            checkpoint.model_dump(mode="json") if checkpoint is not None else None
        )
        _atomic_text(
            json_path,
            json.dumps(machine_payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        return markdown_path, json_path


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
