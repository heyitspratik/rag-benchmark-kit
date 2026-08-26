import pytest

from rag_bench.components.retrievers.dense import DenseRetriever
from rag_bench.components.stores.qdrant import QdrantStore
from rag_bench.core.exceptions import ConfigValidationError, IndexNotReadyError

from .conftest import FixedVectorEmbedder


def test_the_nearest_chunk_comes_first(store: QdrantStore) -> None:
    embedder = FixedVectorEmbedder({"what is pseudonymisation": [0.0, 1.0, 0.0]})

    hits = DenseRetriever(embedder, store, top_k=3).retrieve("what is pseudonymisation")

    assert hits[0].chunk.section_refs == ("GDPR pseudonymisation",)


def test_ranks_are_contiguous_from_zero(store: QdrantStore) -> None:
    hits = DenseRetriever(FixedVectorEmbedder(), store, top_k=3).retrieve("anything")

    assert [hit.rank for hit in hits] == [0, 1, 2]


def test_top_k_bounds_the_result_count(store: QdrantStore) -> None:
    assert len(DenseRetriever(FixedVectorEmbedder(), store, top_k=2).retrieve("q")) == 2


def test_the_call_can_override_top_k(store: QdrantStore) -> None:
    retriever = DenseRetriever(FixedVectorEmbedder(), store, top_k=1)

    assert len(retriever.retrieve("q", k=3)) == 3


def test_querying_an_unbuilt_index_is_reported() -> None:
    from rag_bench.components.stores.qdrant import MEMORY_URL

    store = QdrantStore(collection="never_built", url=MEMORY_URL)
    try:
        with pytest.raises(IndexNotReadyError):
            DenseRetriever(FixedVectorEmbedder(), store).retrieve("q")
    finally:
        store.close()


@pytest.mark.parametrize("top_k", [0, -1])
def test_a_non_positive_top_k_is_rejected(store: QdrantStore, top_k: int) -> None:
    with pytest.raises(ConfigValidationError, match="top_k must be positive"):
        DenseRetriever(FixedVectorEmbedder(), store, top_k=top_k)
