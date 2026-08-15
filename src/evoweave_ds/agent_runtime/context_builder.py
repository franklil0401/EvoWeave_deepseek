"""Build minimal worker context from task specification and artifacts."""

from dataclasses import dataclass

from pydantic import Field

from evoweave_ds.domain.agent_execution_spec import AgentExecutionSpec
from evoweave_ds.domain.base import DomainModel
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import ArtifactId
from evoweave_ds.domain.ports import ArtifactStore


class ContextPolicy(DomainModel):
    max_text_chars: int = Field(default=100_000, ge=1)
    max_artifact_bytes: int = Field(default=10 * 1024 * 1024, ge=1)


@dataclass(frozen=True, slots=True)
class ContextBundle:
    text: str
    included_artifact_ids: tuple[ArtifactId, ...]


class ContextBuilder:
    def __init__(
        self,
        artifact_store: ArtifactStore,
        policy: ContextPolicy | None = None,
    ) -> None:
        self._artifact_store = artifact_store
        self._policy = policy or ContextPolicy()

    def build(self, spec: AgentExecutionSpec) -> ContextBundle:
        sections = [
            f"任务目标：{spec.goal}",
            "验收条件：\n- " + "\n- ".join(spec.acceptance_criteria),
            "已授权能力：" + ", ".join(spec.tool_names),
        ]
        included: list[ArtifactId] = []
        total_artifact_bytes = 0
        artifact_ids = spec.context_artifact_ids + spec.input_artifact_ids
        for artifact_id in artifact_ids:
            ref = self._artifact_store.get_ref(artifact_id)
            data = self._artifact_store.get_bytes(artifact_id)
            total_artifact_bytes += len(data)
            if total_artifact_bytes > self._policy.max_artifact_bytes:
                raise DomainError(
                    ErrorCode.CONTEXT_LIMIT_EXCEEDED,
                    "上下文产物字节数超过上限",
                )
            if ref.media_type.startswith("text/") or ref.media_type == "application/json":
                try:
                    content = data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise DomainError(
                        ErrorCode.CONTEXT_LIMIT_EXCEEDED,
                        f"文本产物不是有效 UTF-8：{artifact_id}",
                    ) from exc
                sections.append(f"产物 {artifact_id}：\n{content}")
                included.append(artifact_id)
            else:
                sections.append(
                    f"二进制产物引用：{artifact_id}，MIME={ref.media_type}，SHA256={ref.sha256}"
                )
                included.append(artifact_id)

        text = "\n\n".join(sections)
        if len(text) > self._policy.max_text_chars:
            raise DomainError(ErrorCode.CONTEXT_LIMIT_EXCEEDED, "文本上下文字符数超过上限")
        return ContextBundle(
            text=text,
            included_artifact_ids=tuple(included),
        )
