"""Compose a cacheable profile from one immutable Git commit."""

from evoweave_ds.domain.artifacts import ArtifactRef
from evoweave_ds.domain.enums import ArtifactKind
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import EvidenceId
from evoweave_ds.domain.ports import ArtifactStore
from evoweave_ds.domain.repository_models import (
    PythonImport,
    PythonSymbol,
    RepositoryAnalysisPolicy,
    RepositoryEvidence,
    RepositoryParseIssue,
    RepositoryProfile,
)
from evoweave_ds.repository.baseline_runner import BaselineRunner
from evoweave_ds.repository.dependency_graph import DependencyGraphBuilder
from evoweave_ds.repository.evidence_builder import EvidenceBuilder, deterministic_evidence_id
from evoweave_ds.repository.file_inventory import FileInventoryBuilder
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.repository.profile_cache import (
    RepositoryProfileCache,
    calculate_profile_digest,
    serialize_profile,
)
from evoweave_ds.repository.python_symbols import PythonSymbolExtractor
from evoweave_ds.repository.test_discovery import ValidationCommandDiscoverer

ANALYZER_VERSION = "2.0"


class RepositoryProfiler:
    def __init__(
        self,
        *,
        cache: RepositoryProfileCache | None = None,
        baseline_runner: BaselineRunner | None = None,
        policy: RepositoryAnalysisPolicy | None = None,
    ) -> None:
        self._cache = cache
        self._baseline_runner = baseline_runner
        self._policy = policy or RepositoryAnalysisPolicy()
        self._inventory = FileInventoryBuilder()
        self._extractor = PythonSymbolExtractor()
        self._dependency_builder = DependencyGraphBuilder()
        self._command_discoverer = ValidationCommandDiscoverer()
        self._evidence_builder = EvidenceBuilder()

    def build(self, inspector: GitInspector) -> RepositoryProfile:
        root = str(inspector.repository_root)
        if self._cache is not None:
            cached = self._cache.get(
                repository_root=root,
                base_commit=inspector.base_commit,
                analyzer_version=ANALYZER_VERSION,
            )
            if cached is not None:
                return cached

        blobs = inspector.list_blobs()
        regular_blobs = tuple(blob for blob in blobs if blob.is_regular_file)
        if len(regular_blobs) > self._policy.max_files:
            raise DomainError(
                ErrorCode.REPOSITORY_LIMIT_EXCEEDED,
                "仓库文件数量超过画像策略上限",
                details={"files": len(regular_blobs), "limit": self._policy.max_files},
            )
        total_bytes = sum(blob.size_bytes for blob in regular_blobs)
        if total_bytes > self._policy.max_total_bytes:
            raise DomainError(
                ErrorCode.REPOSITORY_LIMIT_EXCEEDED,
                "仓库内容大小超过画像策略上限",
                details={"bytes": total_bytes, "limit": self._policy.max_total_bytes},
            )
        inspector.preload_blobs(blobs)
        files = self._inventory.build(inspector, blobs)
        symbols: list[PythonSymbol] = []
        imports: list[PythonImport] = []
        issues: list[RepositoryParseIssue] = []
        evidence: dict[EvidenceId, RepositoryEvidence] = {}

        for file in files:
            file_evidence = self._evidence_builder.source(
                inspector=inspector,
                path=file.path,
                summary=f"固定 commit 中的文件 {file.path}",
            )
            evidence[file_evidence.evidence_id] = file_evidence
            if file.language != "python":
                continue
            if file.size_bytes > self._policy.max_python_file_bytes:
                issue = RepositoryParseIssue(
                    path=file.path,
                    message="Python 文件超过 AST 分析大小上限",
                    evidence_id=deterministic_evidence_id(
                        inspector.base_commit, file.path, "python-size-limit"
                    ),
                )
                issues.append(issue)
                item = self._evidence_builder.source(
                    inspector=inspector,
                    path=file.path,
                    summary=issue.message,
                    evidence_id=issue.evidence_id,
                )
                evidence[item.evidence_id] = item
                continue
            analysis = self._extractor.analyze(
                path=file.path,
                content=inspector.read_bytes(file.path),
                base_commit=inspector.base_commit,
            )
            symbols.extend(analysis.symbols)
            imports.extend(analysis.imports)
            issues.extend(analysis.issues)
            for symbol in analysis.symbols:
                item = self._evidence_builder.source(
                    inspector=inspector,
                    path=symbol.path,
                    summary=f"Python {symbol.kind}：{symbol.qualified_name}",
                    line_start=symbol.line_start,
                    line_end=symbol.line_end,
                    symbol=symbol.qualified_name,
                    evidence_id=symbol.evidence_id,
                )
                evidence[item.evidence_id] = item
            for imported in analysis.imports:
                item = self._evidence_builder.source(
                    inspector=inspector,
                    path=imported.path,
                    summary=f"import：{imported.imported_module}",
                    line_start=imported.line,
                    line_end=imported.line,
                    evidence_id=imported.evidence_id,
                )
                evidence[item.evidence_id] = item
            for issue in analysis.issues:
                item = self._evidence_builder.source(
                    inspector=inspector,
                    path=issue.path,
                    summary=issue.message,
                    line_start=issue.line,
                    line_end=issue.line,
                    evidence_id=issue.evidence_id,
                )
                evidence[item.evidence_id] = item

        symbol_items = tuple(sorted(symbols, key=lambda item: (item.path, item.line_start)))
        import_items = tuple(
            sorted(imports, key=lambda item: (item.path, item.line, item.imported_module))
        )
        issue_items = tuple(sorted(issues, key=lambda item: (item.path, item.line or 0)))
        dependencies = self._dependency_builder.build(files=files, imports=import_items)
        commands = self._command_discoverer.discover(inspector=inspector, files=files)
        baseline_results = (
            self._baseline_runner.run(base_commit=inspector.base_commit, commands=commands)
            if self._baseline_runner is not None
            else ()
        )
        for result in baseline_results:
            item = self._evidence_builder.command(
                base_commit=inspector.base_commit,
                command_id=result.command.command_id,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=result.timed_out,
            )
            evidence[item.evidence_id] = item

        profile = RepositoryProfile(
            analyzer_version=ANALYZER_VERSION,
            base_commit=inspector.base_commit,
            files=files,
            symbols=symbol_items,
            imports=import_items,
            dependencies=dependencies,
            validation_commands=commands,
            baseline_results=baseline_results,
            parse_issues=issue_items,
            evidence=tuple(sorted(evidence.values(), key=lambda item: str(item.evidence_id))),
            profile_digest="0" * 64,
        )
        profile = profile.model_copy(update={"profile_digest": calculate_profile_digest(profile)})
        if self._cache is not None:
            self._cache.put(repository_root=root, profile=profile)
        return profile


def persist_repository_profile(
    profile: RepositoryProfile,
    artifact_store: ArtifactStore,
) -> ArtifactRef:
    """Persist a complete profile behind a small content-addressed artifact reference."""

    return artifact_store.put_bytes(
        serialize_profile(profile),
        media_type="application/json",
        kind=ArtifactKind.REPOSITORY_PROFILE,
    )
