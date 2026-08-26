from unittest.mock import patch

import pytest

from rag_bench.components.retrievers.hybrid import HybridRetriever
from rag_bench.components.stores.qdrant import QdrantStore
from rag_bench.core.exceptions import ConfigValidationError, IndexNotReadyError

from .conftest import FixedVectorEmbedder

# Points the dense half at the supervisory authority chunk, so anything the lexical half
# contributes is unmistakably its own doing.
_AWAY_FROM_FEES = {"reasonable fee": [0.0, 0.0, 1.0]}


def _retriever(store: QdrantStore, **overrides: object) -> HybridRetriever:
    params: dict[str, object] = {
        "embedder": FixedVectorEmbedder(_AWAY_FROM_FEES),
        "store": store,
        "top_k": 3,
    }
    return HybridRetriever(**(params | overrides))  # type: ignore[arg-type]


def test_lexical_matching_recovers_what_dense_search_missed(store: QdrantStore) -> None:
    hits = _retriever(store).retrieve("reasonable fee")

    refs = [hit.chunk.section_refs[0] for hit in hits]
    assert "GDPR fees" in refs


def test_dense_only_weighting_drops_the_lexical_match(store: QdrantStore) -> None:
    hits = _retriever(store, bm25_weight=0.0, top_k=1).retrieve("reasonable fee")

    assert hits[0].chunk.section_refs == ("GDPR authority",)


def test_lexical_only_weighting_finds_the_keyword(store: QdrantStore) -> None:
    hits = _retriever(store, dense_weight=0.0, top_k=1).retrieve("reasonable fee")

    assert hits[0].chunk.section_refs == ("GDPR fees",)


def test_scores_are_fused_not_raw(store: QdrantStore) -> None:
    # Fused scores are small reciprocal-rank sums, not cosine similarities near one.
    hits = _retriever(store).retrieve("reasonable fee")

    assert all(0.0 < hit.score < 0.1 for hit in hits)


def test_ranks_are_contiguous_from_zero(store: QdrantStore) -> None:
    hits = _retriever(store).retrieve("supervisory authority")

    assert [hit.rank for hit in hits] == [0, 1, 2]


def test_top_k_bounds_the_result_count(store: QdrantStore) -> None:
    assert len(_retriever(store, top_k=2).retrieve("fee")) == 2


def test_the_lexical_index_is_built_once_and_reused(store: QdrantStore) -> None:
    # BM25 needs the whole corpus, so rebuilding it per query would make every question
    # after the first pay for a full scan of the collection.
    retriever = _retriever(store)
    with patch.object(store, "iter_chunks", wraps=store.iter_chunks) as scan:
        retriever.retrieve("fee")
        retriever.retrieve("authority")

    assert scan.call_count == 1


def test_a_query_with_no_usable_tokens_still_returns_dense_results(
    store: QdrantStore,
) -> None:
    hits = _retriever(store).retrieve("!!! ???")

    assert hits


def test_an_empty_collection_is_reported(empty_store: QdrantStore) -> None:
    with pytest.raises(IndexNotReadyError, match="collection is empty"):
        _retriever(empty_store).retrieve("fee")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_k": 0}, "top_k must be positive"),
        ({"overfetch_multiplier": 0}, "overfetch_multiplier must be at least 1"),
        ({"bm25_weight": -0.1}, "must not be negative"),
        ({"bm25_weight": 0.0, "dense_weight": 0.0}, "cannot both be zero"),
    ],
)
def test_invalid_parameters_are_rejected(
    store: QdrantStore, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        _retriever(store, **overrides)
