"""A small, hand-vectored index so retrieval order is fully predictable."""

from collections.abc import Iterator, Sequence

import pytest

from rag_bench.components.stores.qdrant import MEMORY_URL, QdrantStore
from rag_bench.core.interfaces import BaseEmbedder, Vector
from rag_bench.core.models import Chunk

DIMENSION = 3

#: Text and vector per chunk. The vectors are axis-aligned so a query vector picks out
#: exactly one of them, and the wording gives BM25 something real to match on.
CORPUS: tuple[tuple[str, str, Vector], ...] = (
    (
        "fees",
        "The controller may charge a reasonable fee based on administrative costs "
        "for any further copies requested by the data subject.",
        [1.0, 0.0, 0.0],
    ),
    (
        "pseudonymisation",
        "Pseudonymisation means the processing of personal data in such a manner "
        "that it can no longer be attributed to a specific data subject.",
        [0.0, 1.0, 0.0],
    ),
    (
        "authority",
        "Each supervisory authority shall monitor and enforce the application of "
        "this Regulation on its territory.",
        [0.0, 0.0, 1.0],
    ),
)


class FixedVectorEmbedder(BaseEmbedder):
    """Returns a preset vector per query, so dense ranking is exactly controllable."""

    def __init__(self, queries: dict[str, Vector] | None = None) -> None:
        self._queries = queries or {}

    @property
    def dimension(self) -> int:
        return DIMENSION

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> Vector:
        return self._queries.get(text, [1.0, 1.0, 1.0])


def make_chunk(ordinal: int, ref: str, text: str) -> Chunk:
    """Build a chunk carrying a recognisable section ref."""
    return Chunk.create(
        doc_id="gdpr", ordinal=ordinal, text=text, char_start=0, char_end=len(text)
    ).with_sections((f"GDPR {ref}",))


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    """An embedded Qdrant collection holding the three corpus chunks."""
    instance = QdrantStore(collection="retriever_test", url=MEMORY_URL)
    instance.ensure_collection(DIMENSION)
    instance.upsert(
        [make_chunk(i, ref, text) for i, (ref, text, _) in enumerate(CORPUS)],
        [vector for _, _, vector in CORPUS],
    )
    yield instance
    instance.close()


@pytest.fixture
def empty_store() -> Iterator[QdrantStore]:
    """An embedded Qdrant collection that exists but holds nothing."""
    instance = QdrantStore(collection="retriever_empty", url=MEMORY_URL)
    instance.ensure_collection(DIMENSION)
    yield instance
    instance.close()
