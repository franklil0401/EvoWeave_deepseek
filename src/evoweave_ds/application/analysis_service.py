"""Create a run and persist the fixed-commit repository profile."""

from evoweave_ds.application.run_state import JsonRunStateStore
from evoweave_ds.domain.change_spec import ChangeSpec
from evoweave_ds.domain.enums import RunStatus
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.ports import ArtifactStore
from evoweave_ds.domain.repository_models import RepositoryProfile
from evoweave_ds.domain.run_models import RunManifest
from evoweave_ds.repository.git_inspector import GitInspector
from evoweave_ds.repository.profile_builder import RepositoryProfiler, persist_repository_profile


class AnalysisService:
    def __init__(
        self,
        *,
        run_store: JsonRunStateStore,
        artifact_store: ArtifactStore,
    ) -> None:
        self._run_store = run_store
        self._artifact_store = artifact_store

    def analyze(self, change_spec: ChangeSpec) -> tuple[RunManifest, RepositoryProfile]:
        manifest = RunManifest(
            run_id=change_spec.run_id,
            status=RunStatus.INITIALIZED,
            change_spec=change_spec,
        )
        self._run_store.create(manifest)
        try:
            inspector = GitInspector(change_spec.repository, change_spec.base_commit)
            profile = RepositoryProfiler().build(inspector)
            profile_ref = persist_repository_profile(profile, self._artifact_store)
            manifest = self._run_store.transition(
                change_spec.run_id,
                RunStatus.ANALYZED,
                message=(
                    f"仓库画像完成：{len(profile.files)} 个文件、"
                    f"{len(profile.symbols)} 个 Python 符号"
                ),
                repository_profile_artifact_id=profile_ref.artifact_id,
            )
            return manifest, profile
        except DomainError as exc:
            self._run_store.transition(
                change_spec.run_id,
                RunStatus.FAILED,
                message=exc.message,
                error_code=exc.code,
            )
            raise
        except Exception:
            self._run_store.transition(
                change_spec.run_id,
                RunStatus.FAILED,
                message="仓库分析发生未预期错误",
                error_code=ErrorCode.PROFILE_INTEGRITY_ERROR,
            )
            raise
