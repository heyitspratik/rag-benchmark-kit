"""Contract for vector store backends, pipeline stage 4."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from rag_bench.core.interfaces.embedder import Vector
from rag_bench.core.models import Chunk, ScoredChunk


class BaseVectorStore(ABC):
    """Persists chunks with their vectors and answers nearest-neighbour queries."""

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it does not exist, sized for the given vectors.

        Args:
            dimension: Vector length the collection must accept.

        Raises:
            VectorStoreError: If the backend is unreachable or rejects the request.
        """

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        """Insert or replace chunks and their vectors.

        Args:
            chunks: The chunks to store.
            vectors: Vectors positionally aligned with ``chunks``.

        Raises:
            ValueError: If the two sequences differ in length.
            VectorStoreError: If the backend rejects the write.
        """

    @abstractmethod
    def search(self, vector: Vector, k: int) -> list[ScoredChunk]:
        """Return the ``k`` nearest chunks to a query vector, best first.

        Args:
            vector: The query vector.
            k: Maximum number of results.

        Returns:
            Scored chunks ordered by descending similarity.

        Raises:
            IndexNotReadyError: If the collection does not exist or is empty.
        """

    @abstractmethod
    def iter_chunks(self) -> Iterator[Chunk]:
        """Stream every stored chunk.

        Lexical retrievers need the whole corpus to build their index, so this is part of
        the contract rather than a Qdrant-specific extra.

        Yields:
            Each stored chunk, in no guaranteed order.
        """

    @abstractmethod
    def count(self) -> int:
        """Number of chunks currently stored, or zero if the collection is absent."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Drop the collection and everything in it. A no-op if it does not exist."""

    @abstractmethod
    def close(self) -> None:
        """Release any backend connection. Implement as a no-op where there is none."""
