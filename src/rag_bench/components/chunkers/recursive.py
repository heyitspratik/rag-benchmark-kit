"""Recursive character chunking, the common LangChain default."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.interfaces import BaseChunker
from rag_bench.core.models import Chunk, Document
from rag_bench.core.registry import register_chunker


@register_chunker("recursive")
class RecursiveChunker(BaseChunker):
    """Splits on the largest natural boundary that fits: paragraph, then sentence, word.

    This is what most RAG tutorials reach for, which makes it the honest point of
    comparison for anything more elaborate.
    """

    def __init__(
        self,
        max_chars: int = 1200,
        overlap: int = 150,
        separators: list[str] | None = None,
    ) -> None:
        """Initialise the chunker.

        Args:
            max_chars: Target chunk size in characters.
            overlap: Characters repeated between neighbouring chunks.
            separators: Boundaries to try, largest first. Defaults to paragraph,
                line, sentence, word, character.

        Raises:
            ConfigValidationError: If the sizes cannot produce progress.
        """
        if max_chars <= 0:
            raise ConfigValidationError(
                f"chunker.params.max_chars must be positive, got {max_chars}"
            )
        if overlap >= max_chars:
            raise ConfigValidationError(
                f"chunker.params.overlap ({overlap}) must be smaller than max_chars ({max_chars})"
            )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
            keep_separator=True,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document on natural boundaries.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order; empty for a blank document.
        """
        if not document.text.strip():
            return []

        chunks: list[Chunk] = []
        for piece in self._splitter.create_documents([document.text]):
            body = piece.page_content
            if not body.strip():
                continue
            # The splitter reports the offset it cut at, so offsets stay truthful even
            # where the same paragraph appears more than once in the document.
            start = int(piece.metadata.get("start_index", 0))
            chunks.append(
                Chunk.create(
                    doc_id=document.id,
                    ordinal=len(chunks),
                    text=body,
                    char_start=start,
                    char_end=start + len(body),
                )
            )
        return chunks
