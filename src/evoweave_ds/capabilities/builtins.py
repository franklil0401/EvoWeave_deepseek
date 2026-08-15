"""Small, composable file and command capabilities used by any worker instance."""

from difflib import unified_diff

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from evoweave_ds.capabilities.definitions import (
    CapabilityContext,
    CapabilityDefinition,
    CapabilityResult,
)
from evoweave_ds.domain.artifacts import EvidenceRef
from evoweave_ds.domain.enums import ArtifactKind, CapabilityAccess, EvidenceKind, RiskLevel
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import EvidenceId


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class _FileReadArguments(_Arguments):
    path: str = Field(min_length=1, max_length=1_024)


class _FileSearchArguments(_Arguments):
    query: str = Field(min_length=1, max_length=1_000)
    prefix: str | None = Field(default=None, min_length=1, max_length=1_024)
    max_matches: int = Field(default=20, ge=1, le=200)


class _FileWriteArguments(_Arguments):
    path: str = Field(min_length=1, max_length=1_024)
    content: str = Field(max_length=2_000_000)


class _CommandArguments(_Arguments):
    argv: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: int = Field(default=120, ge=1, le=3_600)


class FileReadCapability:
    definition = CapabilityDefinition(
        name="file.read",
        description="读取任务授权范围内的一个文本文件",
        access=CapabilityAccess.READ,
        risk_level=RiskLevel.LOW,
        input_schema=_FileReadArguments.model_json_schema(),
    )

    def invoke(
        self,
        arguments: dict[str, JsonValue],
        context: CapabilityContext,
    ) -> CapabilityResult:
        parsed = _validate(_FileReadArguments, arguments)
        content = context.workspace.read_text(parsed.path)
        evidence = EvidenceRef(
            evidence_id=EvidenceId.new(),
            kind=EvidenceKind.FILE,
            summary=f"已读取 {parsed.path}",
            repository_path=parsed.path,
        )
        return CapabilityResult(
            summary=f"读取文件：{parsed.path}",
            details={"path": parsed.path, "content": content},
            evidence=(evidence,),
        )


class FileSearchCapability:
    definition = CapabilityDefinition(
        name="file.search",
        description="在任务授权范围内搜索文本",
        access=CapabilityAccess.READ,
        risk_level=RiskLevel.LOW,
        input_schema=_FileSearchArguments.model_json_schema(),
    )

    def invoke(
        self,
        arguments: dict[str, JsonValue],
        context: CapabilityContext,
    ) -> CapabilityResult:
        parsed = _validate(_FileSearchArguments, arguments)
        matches: list[JsonValue] = []
        evidence: list[EvidenceRef] = []
        for path in context.workspace.list_paths(parsed.prefix):
            try:
                content = context.workspace.read_text(path)
            except DomainError:
                # 非 UTF-8 或不可读文件不参与文本搜索, 避免单个文件阻断整个搜索
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if parsed.query not in line:
                    continue
                matches.append({"path": path, "line": line_number, "text": line})
                evidence.append(
                    EvidenceRef(
                        evidence_id=EvidenceId.new(),
                        kind=EvidenceKind.FILE,
                        summary=f"在 {path}:{line_number} 找到匹配",
                        repository_path=path,
                        line_start=line_number,
                        line_end=line_number,
                    )
                )
                if len(matches) >= parsed.max_matches:
                    break
            if len(matches) >= parsed.max_matches:
                break
        return CapabilityResult(
            summary=f"搜索到 {len(matches)} 条匹配",
            details={"query": parsed.query, "matches": matches},
            evidence=tuple(evidence),
        )


class FileWriteCapability:
    definition = CapabilityDefinition(
        name="file.write",
        description="写入授权范围内的一个文本文件并生成补丁产物",
        access=CapabilityAccess.WRITE,
        risk_level=RiskLevel.HIGH,
        input_schema=_FileWriteArguments.model_json_schema(),
    )

    def invoke(
        self,
        arguments: dict[str, JsonValue],
        context: CapabilityContext,
    ) -> CapabilityResult:
        parsed = _validate(_FileWriteArguments, arguments)
        paths = context.workspace.list_paths()
        before = context.workspace.read_text(parsed.path) if parsed.path in paths else ""
        changed = before != parsed.content
        context.workspace.write_text(parsed.path, parsed.content)
        patch = "".join(
            unified_diff(
                before.splitlines(keepends=True),
                parsed.content.splitlines(keepends=True),
                fromfile=f"a/{parsed.path}",
                tofile=f"b/{parsed.path}",
            )
        )
        artifact = context.artifact_store.put_bytes(
            patch.encode("utf-8"),
            media_type="text/x-diff",
            kind=ArtifactKind.PATCH,
        )
        evidence = EvidenceRef(
            evidence_id=EvidenceId.new(),
            kind=EvidenceKind.ARTIFACT,
            summary=f"已生成 {parsed.path} 的补丁",
            artifact_id=artifact.artifact_id,
        )
        return CapabilityResult(
            summary=(f"已写入 {parsed.path}" if changed else f"{parsed.path} 内容没有变化"),
            details={
                "path": parsed.path,
                "artifact_id": str(artifact.artifact_id),
                "changed": changed,
            },
            evidence=(evidence,),
            artifacts=(artifact,),
        )


class CommandRunCapability:
    definition = CapabilityDefinition(
        name="command.run",
        description="通过无 shell 的命令适配器运行授权命令",
        access=CapabilityAccess.COMMAND,
        risk_level=RiskLevel.HIGH,
        input_schema=_CommandArguments.model_json_schema(),
    )

    def invoke(
        self,
        arguments: dict[str, JsonValue],
        context: CapabilityContext,
    ) -> CapabilityResult:
        parsed = _validate(_CommandArguments, arguments)
        if context.command_runner is None:
            raise DomainError(ErrorCode.COMMAND_DENIED, "未配置命令执行适配器")
        result = context.command_runner.run(
            parsed.argv,
            timeout_seconds=parsed.timeout_seconds,
        )
        log = (
            f"argv={list(result.argv)!r}\n"
            f"exit_code={result.exit_code}\n"
            f"timed_out={result.timed_out}\n"
            f"duration_ms={result.duration_ms}\n"
            f"output_truncated={result.output_truncated}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )
        artifact = context.artifact_store.put_bytes(
            log.encode("utf-8"),
            media_type="text/plain",
            kind=ArtifactKind.COMMAND_LOG,
        )
        evidence = EvidenceRef(
            evidence_id=EvidenceId.new(),
            kind=EvidenceKind.COMMAND,
            summary=f"命令退出码：{result.exit_code}",
            artifact_id=artifact.artifact_id,
            command=" ".join(result.argv),
        )
        return CapabilityResult(
            summary=f"命令执行完成，退出码 {result.exit_code}",
            details={
                "argv": list(result.argv),
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_ms": result.duration_ms,
                "output_truncated": result.output_truncated,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            evidence=(evidence,),
            artifacts=(artifact,),
        )


def default_capabilities() -> tuple[
    FileReadCapability,
    FileSearchCapability,
    FileWriteCapability,
    CommandRunCapability,
]:
    return (
        FileReadCapability(),
        FileSearchCapability(),
        FileWriteCapability(),
        CommandRunCapability(),
    )


def _validate[ArgumentsT: _Arguments](
    model: type[ArgumentsT],
    arguments: dict[str, JsonValue],
) -> ArgumentsT:
    try:
        return model.model_validate(arguments)
    except ValidationError as exc:
        raise DomainError(
            ErrorCode.INVALID_SPEC,
            "能力参数不符合输入协议",
            details={"errors": exc.errors(include_url=False)},
        ) from exc
