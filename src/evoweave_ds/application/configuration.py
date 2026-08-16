"""Versioned non-secret runtime configuration loaded from JSON-compatible YAML."""

import json
from pathlib import Path

from pydantic import Field, field_validator

from evoweave_ds.domain.base import DomainModel


class EvoWeaveConfig(DomainModel):
    runtime_directory: str = Field(default=".runtime", min_length=1, max_length=255)
    default_provider: str = Field(default="deepseek", pattern=r"^[a-z0-9_-]+$")
    default_model_id: str = Field(default="deepseek-v4-flash", min_length=1, max_length=255)
    sandbox_image: str = Field(default="evoweave_ds-python:3.12", min_length=1, max_length=512)
    max_worker_steps: int = Field(default=32, ge=1, le=1_000)
    max_worker_tool_calls: int = Field(default=32, ge=1, le=1_000)
    max_worker_seconds: int = Field(default=900, ge=1, le=86_400)
    max_dynamic_tasks: int = Field(default=8, ge=1, le=128)
    split_directory_lines: int = Field(default=400, ge=1, le=1_000_000)
    # 借鉴 dsh 可续接子代理: 失败 Worker 带上下文续接重试(默认关闭,
    # 保持既有实验行为; 开启后失败任务复用上一执行规格派生下一版本)。
    worker_continuation: bool = False

    @field_validator("runtime_directory")
    @classmethod
    def validate_runtime_directory(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("runtime_directory 必须是仓库内相对路径")
        return value


def load_config(path: Path | str | None = None) -> EvoWeaveConfig:
    if path is None:
        return EvoWeaveConfig()
    config_path = Path(path).resolve(strict=True)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("配置必须是 JSON 兼容 YAML") from exc
    if not isinstance(payload, dict):
        raise ValueError("配置顶层必须是对象")
    return EvoWeaveConfig.model_validate(payload)
