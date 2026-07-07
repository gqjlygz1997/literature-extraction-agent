"""Retrieve candidate evidence chunks from parsed papers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalQuery:
    """A semantic query generated for one target field."""

    field_name: str
    query: str
    top_k: int = 5


@dataclass(frozen=True)
class CandidateChunk:
    """A retrieved paragraph or table candidate."""

    paper_id: str
    chunk_id: str
    text: str
    score: float | None = None
    source: str = "semantic"


class Retriever:
    """Evidence retriever interface."""

    def retrieve(self, chunks, query: RetrievalQuery) -> list[CandidateChunk]:
        raise NotImplementedError("Semantic or hybrid retrieval is not implemented yet.")

