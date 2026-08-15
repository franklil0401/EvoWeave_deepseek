"""Content-validated in-memory cache for deterministic repository profiles."""

import json
from hashlib import sha256

from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.repository_models import RepositoryProfile


def calculate_profile_digest(profile: RepositoryProfile) -> str:
    payload = profile.model_dump(mode="json", exclude={"profile_digest"})
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def serialize_profile(profile: RepositoryProfile) -> bytes:
    _verify_digest(profile)
    return profile.model_dump_json().encode("utf-8")


class RepositoryProfileCache:
    def __init__(self) -> None:
        self._profiles: dict[tuple[str, str, str], bytes] = {}

    def get(
        self,
        *,
        repository_root: str,
        base_commit: str,
        analyzer_version: str,
    ) -> RepositoryProfile | None:
        data = self._profiles.get((repository_root, base_commit, analyzer_version))
        if data is None:
            return None
        profile = RepositoryProfile.model_validate_json(data)
        _verify_digest(profile)
        return profile

    def put(self, *, repository_root: str, profile: RepositoryProfile) -> None:
        self._profiles[(repository_root, profile.base_commit, profile.analyzer_version)] = (
            serialize_profile(profile)
        )


def _verify_digest(profile: RepositoryProfile) -> None:
    actual = calculate_profile_digest(profile)
    if actual != profile.profile_digest:
        raise DomainError(
            ErrorCode.PROFILE_INTEGRITY_ERROR,
            "仓库画像摘要校验失败",
            details={"expected": profile.profile_digest, "actual": actual},
        )
