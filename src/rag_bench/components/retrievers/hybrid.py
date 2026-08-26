"""Hybrid retrieval: lexical BM25 and dense vectors, merged by rank."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rag_bench.components.retrievers.fusion import (
    DEFAULT_RRF_K,
    RankedList,
    ReciprocalRankFusion,
)
from rag_bench.core.exceptions import ConfigValidationError, IndexNotReadyError
from rag_bench.core.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Chunk, ScoredChunk
from rag_bench.core.registry import register_retriever

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@register_retriever("hybrid")
class HybridRetriever(BaseRetriever):
    """Runs BM25 and vector search over the same corpus and fuses the two rankings.

    The two find different things. Dense search handles paraphrase, so a question about
    "turning someone down" reaches an article that says "refuse". BM25 handles exact
    terms, so "Article 15(4)" or "pseudonymisation" is found verbatim rather than
    approximately. Regulatory text is full of both, which is why neither alone is enough.
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
    ) -> None:
        """Initialise the retriever.

        Args:
            embedder: Query embedder, injected by the querier.
            store: The vector store holding the built index.
            top_k: Default number of chunks to return.
            bm25_weight: Weight given to the lexical ranking.
            dense_weight: Weight given to the vector ranking.
            rrf_k: Reciprocal Rank Fusion damping constant.
            overfetch_multiplier: How many candidates each ranker contributes, as a
                multiple of ``top_k``. Fusing only the final k from each would discard
                the very results the other ranker is there to recover.

        Raises:
            ConfigValidationError: If any bound is invalid, or both weights are zero.
        """
        if top_k <= 0:
            raise ConfigValidationError(f"retriever.params.top_k must be positive, got {top_k}")
        if overfetch_multiplier < 1:
            raise ConfigValidationError(
                "retriever.params.overfetch_multiplier must be at least 1, "
                f"got {overfetch_multiplier}"
            )
        if bm25_weight < 0.0 or dense_weight < 0.0:
            raise ConfigValidationError("retriever fusion weights must not be negative")
        if bm25_weight == 0.0 and dense_weight == 0.0:
            raise ConfigValidationError(
                "retriever.params.bm25_weight and dense_weight cannot both be zero"
            )

        self._embedder = embedder
        self._store = store
        self.top_k = top_k
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight
        self._overfetch = overfetch_multiplier
        self._fusion = ReciprocalRankFusion(rrf_k)
        self._bm25: BM25Okapi | None = None
        self._corpus: list[Chunk] = []

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredChunk]:
        """Retrieve by both methods and fuse the rankings.

        Args:
            query: The user's question.
            k: How many to return; falls back to the configured ``top_k``.

        Returns:
            Scored chunks ordered best first, carrying fused scores rather than either
            ranker's raw score.

        Raises:
            IndexNotReadyError: If no index has been built.
        """
        wanted = k or self.top_k
        candidates = wanted * self._overfetch

        dense_hits = self._store.search(self._embedder.embed_query(query), candidates)
        lexical_hits = self._lexical(query, candidates)

        by_id: dict[str, Chunk] = {hit.chunk.id: hit.chunk for hit in dense_hits}
        by_id.update({chunk.id: chunk for chunk in lexical_hits})

        fused = self._fusion.fuse(
            [
                RankedList("dense", self._dense_weight, tuple(h.chunk.id for h in dense_hits)),
                RankedList("bm25", self._bm25_weight, tuple(c.id for c in lexical_hits)),
            ]
        )
        return [
            ScoredChunk(chunk=by_id[result.id], score=result.score, rank=rank)
            for rank, result in enumerate(fused[:wanted])
        ]

    def _lexical(self, query: str, limit: int) -> list[Chunk]:
        """Return the best BM25 matches for a query, best first."""
        bm25, corpus = self._load_corpus()
        if not corpus:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        ordered = sorted(range(len(corpus)), key=lambda i: (-scores[i], corpus[i].id))
        return [corpus[i] for i in ordered[:limit] if scores[i] > 0.0]

    def _load_corpus(self) -> tuple[BM25Okapi, list[Chunk]]:
        """Build the BM25 index on first use.

        BM25 needs the whole corpus in memory, so this is deferred until a query actually
        arrives and then kept for the life of the retriever. That is affordable at the
        scale this benchmark works at and would need revisiting well before it is not.

        Raises:
            IndexNotReadyError: If the collection holds nothing to search.
        """
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._corpus = list(self._store.iter_chunks())
            if not self._corpus:
                raise IndexNotReadyError(
                    "The collection is empty. Run `rag-bench index build` first."
                )
            logger.info("retriever.bm25_built", chunks=len(self._corpus))
            self._bm25 = BM25Okapi([_tokenize(chunk.text) for chunk in self._corpus])
        return self._bm25, self._corpus


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, which is all BM25 needs."""
    return _TOKEN_RE.findall(text.lower())
