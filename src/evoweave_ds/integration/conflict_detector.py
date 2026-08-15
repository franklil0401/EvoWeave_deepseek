"""Preflight actual write-set conflicts before any patch is applied."""

from evoweave_ds.domain.enums import PatchConflictKind
from evoweave_ds.domain.integration_models import GuardedPatch, PatchConflict


class PatchConflictDetector:
    def detect(self, patches: tuple[GuardedPatch, ...]) -> tuple[PatchConflict, ...]:
        conflicts: list[PatchConflict] = []
        for index, first in enumerate(patches):
            first_paths = set(first.parsed_paths)
            for second in patches[index + 1 :]:
                overlap = tuple(sorted(first_paths.intersection(second.parsed_paths)))
                if overlap:
                    conflicts.append(
                        PatchConflict(
                            kind=PatchConflictKind.WRITE_SET,
                            message="两个补丁修改了同一实际路径",
                            artifact_ids=(
                                first.artifact.ref.artifact_id,
                                second.artifact.ref.artifact_id,
                            ),
                            paths=overlap,
                        )
                    )
        return tuple(conflicts)
