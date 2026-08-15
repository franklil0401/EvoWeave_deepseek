"""Durable content-addressed artifact storage with atomic metadata writes."""

import json
import os
from hashlib import sha256
from pathlib import Path
from threading import RLock

from evoweave_ds.domain.artifacts import ArtifactRef
from evoweave_ds.domain.enums import ArtifactKind
from evoweave_ds.domain.errors import DomainError, ErrorCode
from evoweave_ds.domain.identifiers import ArtifactId


class LocalArtifactStore:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._content_root = self._root / "sha256"
        self._ref_root = self._root / "refs"
        self._content_root.mkdir(parents=True, exist_ok=True)
        self._ref_root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        kind: ArtifactKind,
    ) -> ArtifactRef:
        digest = sha256(data).hexdigest()
        with self._lock:
            existing = self._find_digest(digest)
            if existing is not None:
                if existing.media_type != media_type or existing.kind is not kind:
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_ERROR,
                        "相同内容不能登记为不同产物类型",
                    )
                return existing
            content_path = self._content_path(digest)
            content_path.parent.mkdir(parents=True, exist_ok=True)
            if content_path.exists():
                persisted = content_path.read_bytes()
                if sha256(persisted).hexdigest() != digest:
                    raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "内容存储已损坏")
            else:
                _atomic_write(content_path, data)
            artifact_id = ArtifactId.new()
            reference = ArtifactRef(
                artifact_id=artifact_id,
                kind=kind,
                media_type=media_type,
                size_bytes=len(data),
                sha256=digest,
                storage_key=f"sha256/{digest[:2]}/{digest}",
            )
            self._write_ref(reference)
            return reference

    def get_bytes(self, artifact_id: ArtifactId) -> bytes:
        reference = self.get_ref(artifact_id)
        try:
            data = self._content_path(reference.sha256).read_bytes()
        except FileNotFoundError as exc:
            raise DomainError(ErrorCode.ARTIFACT_NOT_FOUND, "产物内容不存在") from exc
        if len(data) != reference.size_bytes or sha256(data).hexdigest() != reference.sha256:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "产物内容校验失败")
        return data

    def get_ref(self, artifact_id: ArtifactId) -> ArtifactRef:
        path = self._ref_path(artifact_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DomainError(ErrorCode.ARTIFACT_NOT_FOUND, "产物引用不存在") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "产物引用无法解析") from exc
        try:
            return ArtifactRef.model_validate(payload)
        except ValueError as exc:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "产物引用校验失败") from exc

    def update_ref(self, ref: ArtifactRef) -> None:
        with self._lock:
            existing = self.get_ref(ref.artifact_id)
            immutable = (
                "artifact_id",
                "kind",
                "media_type",
                "size_bytes",
                "sha256",
                "storage_key",
                "created_at",
            )
            if any(getattr(existing, name) != getattr(ref, name) for name in immutable):
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_ERROR,
                    "不能修改产物内容身份字段",
                )
            self._write_ref(ref)

    def _find_digest(self, digest: str) -> ArtifactRef | None:
        for path in self._ref_root.glob("artifact_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if payload.get("sha256") == digest:
                return self.get_ref(ArtifactId(payload["artifact_id"]))
        return None

    def _write_ref(self, ref: ArtifactRef) -> None:
        payload = ref.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _atomic_write(self._ref_path(ref.artifact_id), encoded)

    def _content_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "产物摘要无效")
        return self._content_root / digest[:2] / digest

    def _ref_path(self, artifact_id: ArtifactId) -> Path:
        path = (self._ref_root / f"{artifact_id}.json").resolve()
        if path.parent != self._ref_root:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_ERROR, "产物引用路径越界")
        return path


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
