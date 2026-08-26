"""Fixed-size chunking: the naive baseline every other strategy is measured against."""

from __future__ import annotations

from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.interfaces import BaseChunker
from rag_bench.core.models import Chunk, Document
from rag_bench.core.registry import register_chunker


@register_chunker("fixed")
class FixedChunker(BaseChunker):
    """Slices a document into equal character windows with a fixed overlap.

    It knows nothing about sentences, paragraphs, or articles, so it routinely cuts a
    definition away from the sentence that uses it. That is the point: it is the baseline
    the benchmark exists to beat.
    """

    def __init__(self, max_chars: int = 1200, overlap: int = 150) -> None:
        """Initialise the chunker.

        Args:
            max_chars: Window size in characters.
            overlap: Characters each window repeats from the previous one.

        Raises:
            ConfigValidationError: If the window would not advance.
        """
        if max_chars <= 0:
            raise ConfigValidationError(
                f"chunker.params.max_chars must be positive, got {max_chars}"
            )
        if overlap < 0:
            raise ConfigValidationError(
                f"chunker.params.overlap must not be negative, got {overlap}"
            )
        if overlap >= max_chars:
            raise ConfigValidationError(
                f"chunker.params.overlap ({overlap}) must be smaller than max_chars "
                f"({max_chars}), otherwise the window never advances"
            )
        self._max_chars = max_chars
        self._stride = max_chars - overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into fixed-size overlapping windows.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order; empty for a blank document.
        """
        text = document.text
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self._max_chars, len(text))
            body = text[start:end]
            if body.strip():
                chunks.append(
                    Chunk.create(
                        doc_id=document.id,
                        ordinal=len(chunks),
                        text=body,
                        char_start=start,
                        char_end=end,
                    )
                )
            if end == len(text):
                break
            start += self._stride
        return chunks
