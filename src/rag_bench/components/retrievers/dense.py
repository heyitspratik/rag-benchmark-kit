"""Dense retrieval: nearest neighbours in embedding space, and nothing else."""

from __future__ import annotations

from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore
from rag_bench.core.models import ScoredChunk
from rag_bench.core.registry import register_retriever


@register_retriever("dense")
class DenseRetriever(BaseRetriever):
    """Embeds the query and returns its nearest chunks.

    The baseline. It finds paraphrases that keyword search misses, and misses exact terms
    that keyword search finds, which is the whole reason the hybrid variants exist.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        top_k: int = 5,
    ) -> None:
        """Initialise the retriever.

        Args:
            embedder: Query embedder, injected by the querier so that the same model
                indexes and searches.
            store: The vector store holding the built index.
            top_k: Default number of chunks to return.

        Raises:
            ConfigValidationError: If ``top_k`` is not positive.
        """
        if top_k <= 0:
            raise ConfigValidationError(f"retriever.params.top_k must be positive, got {top_k}")
        self._embedder = embedder
        self._store = store
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredChunk]:
        """Return the chunks nearest the query, best first.

        Args:
            query: The user's question.
            k: How many to return; falls back to the configured ``top_k``.

        Returns:
            Scored chunks ordered best first.

        Raises:
            IndexNotReadyError: If no index has been built.
        """
        return self._store.search(self._embedder.embed_query(query), k or self.top_k)
