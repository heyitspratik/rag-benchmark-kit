"""Hybrid retrieval followed by a cross-encoder rerank."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_bench.components.retrievers.fusion import DEFAULT_RRF_K
from rag_bench.components.retrievers.hybrid import HybridRetriever
from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore
from rag_bench.core.logging import get_logger
from rag_bench.core.models import ScoredChunk
from rag_bench.core.registry import register_retriever

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


@register_retriever("hybrid_rerank")
class HybridRerankRetriever(BaseRetriever):
    """Over-fetches with hybrid retrieval, then reorders with a cross-encoder.

    A bi-encoder embeds the query and the chunk separately, so it never compares them
    directly. A cross-encoder reads both together and scores the pair, which is far more
    accurate and far too slow to run over a whole corpus. Using the first to shortlist
    and the second to reorder is what makes the accuracy affordable.

    Composed from :class:`HybridRetriever` rather than inheriting from it: reranking is a
    stage applied to someone else's results, not a different way of retrieving.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        top_k: int = 5,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        rrf_k: int = DEFAULT_RRF_K,
        overfetch_multiplier: int = 4,
        reranker_model: str = "BAAI/bge-reranker-base",
        rerank_batch_size: int = 32,
    ) -> None:
        """Initialise the retriever.

        Args:
            embedder: Query embedder, injected by the querier.
            store: The vector store holding the built index.
            top_k: Number of chunks to return after reranking.
            bm25_weight: Weight given to the lexical ranking.
            dense_weight: Weight given to the vector ranking.
            rrf_k: Reciprocal Rank Fusion damping constant.
            overfetch_multiplier: Shortlist size, as a multiple of ``top_k``.
            reranker_model: Cross-encoder to score query and chunk together. The default
                runs locally on CPU and needs no key.
            rerank_batch_size: Pairs scored per forward pass.

        Raises:
            ConfigValidationError: If any bound is invalid.
        """
        if top_k <= 0:
            raise ConfigValidationError(f"retriever.params.top_k must be positive, got {top_k}")
        if overfetch_multiplier < 1:
            raise ConfigValidationError(
                "retriever.params.overfetch_multiplier must be at least 1, "
                f"got {overfetch_multiplier}"
            )
        if rerank_batch_size <= 0:
            raise ConfigValidationError(
                f"retriever.params.rerank_batch_size must be positive, got {rerank_batch_size}"
            )

        self.top_k = top_k
        self._overfetch = overfetch_multiplier
        self._reranker_model = reranker_model
        self._batch_size = rerank_batch_size
        self._reranker: CrossEncoder | None = None
        self._hybrid = HybridRetriever(
            embedder=embedder,
            store=store,
            top_k=top_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k,
            overfetch_multiplier=overfetch_multiplier,
        )

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredChunk]:
        """Shortlist with hybrid retrieval, then rerank down to ``k``.

        Args:
            query: The user's question.
            k: How many to return; falls back to the configured ``top_k``.

        Returns:
            Scored chunks ordered best first, carrying cross-encoder scores.

        Raises:
            IndexNotReadyError: If no index has been built.
        """
        wanted = k or self.top_k
        shortlist = self._hybrid.retrieve(query, wanted * self._overfetch)
        if not shortlist:
            return []

        scores = self._score(query, [hit.chunk.text for hit in shortlist])
        ordered = sorted(
            zip(shortlist, scores, strict=True),
            key=lambda pair: (-pair[1], pair[0].chunk.id),
        )
        return [
            ScoredChunk(chunk=hit.chunk, score=score, rank=rank)
            for rank, (hit, score) in enumerate(ordered[:wanted])
        ]

    def _score(self, query: str, passages: list[str]) -> list[float]:
        """Score every query and passage pair with the cross-encoder."""
        pairs = [(query, passage) for passage in passages]
        predictions = self._load().predict(pairs, batch_size=self._batch_size)
        return [float(value) for value in predictions]

    def _load(self) -> CrossEncoder:
        """Load the cross-encoder on first use, not at construction."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            logger.info("retriever.reranker_loading", model=self._reranker_model)
            self._reranker = CrossEncoder(self._reranker_model)
        return self._reranker
