from unittest.mock import MagicMock, patch

import pytest

from rag_bench.components.retrievers.hybrid_rerank import HybridRerankRetriever
from rag_bench.components.stores.qdrant import QdrantStore
from rag_bench.core.exceptions import ConfigValidationError

from .conftest import FixedVectorEmbedder


def _retriever(store: QdrantStore, **overrides: object) -> HybridRerankRetriever:
    params: dict[str, object] = {
        "embedder": FixedVectorEmbedder(),
        "store": store,
        "top_k": 2,
        "overfetch_multiplier": 2,
    }
    return HybridRerankRetriever(**(params | overrides))  # type: ignore[arg-type]


def _cross_encoder(scores: list[float]) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = scores
    return model


def test_the_cross_encoder_decides_the_final_order(store: QdrantStore) -> None:
    # Reverse whatever the shortlist order was: the last candidate scores highest.
    model = MagicMock()
    model.predict.side_effect = lambda pairs, batch_size: list(  # noqa: ARG005
        range(len(pairs))
    )

    with patch("sentence_transformers.CrossEncoder", return_value=model):
        hits = _retriever(store, top_k=3).retrieve("anything")

    assert [hit.score for hit in hits] == sorted((h.score for h in hits), reverse=True)


def test_the_shortlist_is_larger_than_what_is_returned(store: QdrantStore) -> None:
    model = _cross_encoder([0.1, 0.9, 0.5])

    with patch("sentence_transformers.CrossEncoder", return_value=model):
        hits = _retriever(store, top_k=1, overfetch_multiplier=3).retrieve("anything")

    assert len(model.predict.call_args.args[0]) == 3
    assert len(hits) == 1


def test_the_query_is_paired_with_every_candidate(store: QdrantStore) -> None:
    model = _cross_encoder([0.1, 0.2, 0.3])

    with patch("sentence_transformers.CrossEncoder", return_value=model):
        _retriever(store, top_k=3).retrieve("a legal question")

    pairs = model.predict.call_args.args[0]
    assert all(pair[0] == "a legal question" for pair in pairs)


def test_scores_are_the_cross_encoder_scores(store: QdrantStore) -> None:
    model = _cross_encoder([0.10, 0.90, 0.50])

    with patch("sentence_transformers.CrossEncoder", return_value=model):
        hits = _retriever(store, top_k=3).retrieve("anything")

    assert [hit.score for hit in hits] == [0.90, 0.50, 0.10]


def test_ranks_are_contiguous_from_zero(store: QdrantStore) -> None:
    with patch("sentence_transformers.CrossEncoder", return_value=_cross_encoder([1.0, 2.0, 3.0])):
        hits = _retriever(store, top_k=3).retrieve("anything")

    assert [hit.rank for hit in hits] == [0, 1, 2]


def test_the_reranker_loads_once_and_only_when_used(store: QdrantStore) -> None:
    with patch(
        "sentence_transformers.CrossEncoder", return_value=_cross_encoder([1.0, 2.0, 3.0])
    ) as constructor:
        retriever = _retriever(store, top_k=3)
        assert constructor.call_count == 0

        retriever.retrieve("first")
        retriever.retrieve("second")

    assert constructor.call_count == 1


def test_an_empty_shortlist_skips_the_reranker(store: QdrantStore) -> None:
    retriever = _retriever(store)
    with (
        patch.object(retriever, "_hybrid") as hybrid,
        patch("sentence_transformers.CrossEncoder") as constructor,
    ):
        hybrid.retrieve.return_value = []

        assert retriever.retrieve("anything") == []

    constructor.assert_not_called()


def test_the_configured_model_is_the_one_loaded(store: QdrantStore) -> None:
    with patch(
        "sentence_transformers.CrossEncoder", return_value=_cross_encoder([1.0, 2.0])
    ) as constructor:
        _retriever(store, top_k=1, reranker_model="some/other-reranker").retrieve("q")

    assert constructor.call_args.args[0] == "some/other-reranker"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_k": 0}, "top_k must be positive"),
        ({"overfetch_multiplier": 0}, "overfetch_multiplier must be at least 1"),
        ({"rerank_batch_size": 0}, "rerank_batch_size must be positive"),
    ],
)
def test_invalid_parameters_are_rejected(
    store: QdrantStore, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        _retriever(store, **overrides)
