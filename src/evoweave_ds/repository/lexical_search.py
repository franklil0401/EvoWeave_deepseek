"""Deterministic literal search over text blobs at one commit."""

from evoweave_ds.domain.repository_models import RepositoryFile, RequirementClues, SearchHit
from evoweave_ds.repository.evidence_builder import deterministic_evidence_id
from evoweave_ds.repository.git_inspector import GitInspector


class LexicalSearcher:
    def search(
        self,
        *,
        inspector: GitInspector,
        files: tuple[RepositoryFile, ...],
        clues: RequirementClues,
        max_hits: int = 200,
        max_file_bytes: int = 1_000_000,
    ) -> tuple[SearchHit, ...]:
        terms = tuple(dict.fromkeys((*clues.paths, *clues.symbols, *clues.terms)))
        hits: list[SearchHit] = []
        for file in files:
            if file.size_bytes > max_file_bytes or file.language is None:
                continue
            content = inspector.read_bytes(file.path)
            try:
                lines = content.decode("utf-8-sig").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                folded = line.casefold()
                for term in terms:
                    if term.casefold() not in folded:
                        continue
                    excerpt = line.strip()[:2_000] or "（空白匹配行）"
                    hits.append(
                        SearchHit(
                            path=file.path,
                            line=line_number,
                            term=term,
                            excerpt=excerpt,
                            evidence_id=deterministic_evidence_id(
                                inspector.base_commit,
                                file.path,
                                line_number,
                                term,
                                excerpt,
                            ),
                        )
                    )
                    if len(hits) >= max_hits:
                        return tuple(hits)
        return tuple(hits)
