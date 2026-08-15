from evoweave_ds.domain.enums import ArtifactKind
from evoweave_ds.infrastructure.artifacts.local_store import LocalArtifactStore


def test_local_store_persists_content_and_reference(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    reference = store.put_bytes(
        b"content",
        media_type="text/plain",
        kind=ArtifactKind.GENERIC,
    )
    restarted = LocalArtifactStore(tmp_path / "artifacts")

    assert restarted.get_bytes(reference.artifact_id) == b"content"
    assert restarted.get_ref(reference.artifact_id) == reference


def test_local_store_deduplicates_equal_content(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = store.put_bytes(b"same", media_type="text/plain", kind=ArtifactKind.GENERIC)
    second = store.put_bytes(b"same", media_type="text/plain", kind=ArtifactKind.GENERIC)

    assert first == second
