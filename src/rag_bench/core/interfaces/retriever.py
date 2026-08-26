"""Contract for retrieval strategies, pipeline stage 5."""

from abc import ABC, abstractmethod

from rag_bench.core.models import ScoredChunk


class BaseRetriever(ABC):
    """Finds the chunks most likely to answer a question."""

    #: Default number of chunks to return, set from config by each implementation.
    top_k: int = 5

    @abstractmethod
    def retrieve(self, query: str, k: int | None = None) -> list[ScoredChunk]:
        """Retrieve chunks for a query, best first.

        Args:
            query: The user's question, unmodified.
            k: How many chunks to return; falls back to the configured ``top_k``.

        Returns:
            Scored chunks ordered best first, with contiguous ranks from zero.

        Raises:
            IndexNotReadyError: If no index has been built for this configuration.
        """
