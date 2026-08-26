"""Contract for chunking strategies, pipeline stage 2."""

from abc import ABC, abstractmethod

from rag_bench.core.models import Chunk, Document


class BaseChunker(ABC):
    """Splits a document into retrievable chunks.

    Implementations must set truthful ``char_start`` / ``char_end`` offsets on every
    chunk. Those offsets are how a chunk is mapped back to the document sections it
    covers, which is what makes retrieval quality comparable between strategies that
    otherwise share no notion of structure.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split one document.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order, each with offsets into ``document.text``. An empty
            or whitespace-only document yields an empty list rather than an error.
        """
