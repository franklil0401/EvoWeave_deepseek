"""Build deterministic evidence locators and verify they against a commit."""

from hashlib import sha256

from evoweave_ds.domain.identifiers import EvidenceId
from evoweave_ds.domain.repository_models import RepositoryEvidence
from evoweave_ds.repository.git_inspector import GitInspector


def deterministic_evidence_id(*parts: object) -> EvidenceId:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return EvidenceId(f"evidence_{sha256(payload).hexdigest()[:24]}")


class EvidenceBuilder:
    def source(
        self,
        *,
        inspector: GitInspector,
        path: str,
        summary: str,
        line_start: int | None = None,
        line_end: int | None = None,
        symbol: str | None = None,
        evidence_id: EvidenceId | None = None,
    ) -> RepositoryEvidence:
        content = inspector.read_bytes(path)
        selected = _select_lines(content, line_start, line_end)
        resolved_evidence_id = evidence_id or deterministic_evidence_id(
            inspector.base_commit, path, line_start, line_end, symbol, sha256(selected).hexdigest()
        )
        return RepositoryEvidence(
            evidence_id=resolved_evidence_id,
            base_commit=inspector.base_commit,
            summary=summary,
            path=path,
            symbol=symbol,
            line_start=line_start,
            line_end=line_end,
            content_sha256=sha256(selected).hexdigest(),
        )

    def command(
        self,
        *,
        base_commit: str,
        command_id: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> RepositoryEvidence:
        payload = f"{command_id}\0{exit_code}\0{timed_out}\0{stdout}\0{stderr}".encode()
        digest = sha256(payload).hexdigest()
        return RepositoryEvidence(
            evidence_id=deterministic_evidence_id(base_commit, "command", command_id, digest),
            base_commit=base_commit,
            summary=f"基线命令 {command_id} 的退出码为 {exit_code}",
            content_sha256=digest,
        )

    def verify(self, evidence: RepositoryEvidence, inspector: GitInspector) -> bool:
        if evidence.base_commit != inspector.base_commit or evidence.path is None:
            return False
        content = inspector.read_bytes(evidence.path)
        selected = _select_lines(content, evidence.line_start, evidence.line_end)
        return sha256(selected).hexdigest() == evidence.content_sha256


def _select_lines(data: bytes, line_start: int | None, line_end: int | None) -> bytes:
    if line_start is None:
        return data
    lines = data.splitlines(keepends=True)
    end = line_end if line_end is not None else line_start
    return b"".join(lines[line_start - 1 : end])
