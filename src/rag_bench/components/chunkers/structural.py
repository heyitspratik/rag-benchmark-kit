"""Structural chunking: split where the document itself says one unit ends."""

from __future__ import annotations

from rag_bench.components.chunkers.spans import SpanBudget
from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.interfaces import BaseChunker
from rag_bench.core.models import Chunk, Document, DocumentSection
from rag_bench.core.registry import register_chunker

_PARAGRAPH_SEPARATOR = "\n\n"
_TOP_LEVEL = 1
_SUB_LEVEL = 2


@register_chunker("structural")
class StructuralChunker(BaseChunker):
    """Emits one chunk per top-level unit, splitting only units that are too long.

    The unit boundaries come from the loader, which is the only component that knows what
    the corpus is: articles for a regulation, headings for Markdown. Keeping the corpus
    knowledge there rather than here is what lets this strategy work unchanged on a corpus
    it has never seen, and it avoids a second copy of the article regex drifting out of
    step with the first.

    A document with no sections falls back to grouping paragraphs, so this never degrades
    into returning the whole document as one chunk.
    """

    def __init__(self, max_chars: int = 1200, overlap: int = 150) -> None:
        """Initialise the chunker.

        Args:
            max_chars: Longest chunk to emit before a unit is subdivided.
            overlap: Characters repeated when a unit has to be subdivided.

        Raises:
            ConfigValidationError: If the sizes cannot produce progress.
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
                f"chunker.params.overlap ({overlap}) must be smaller than max_chars ({max_chars})"
            )
        self._budget = SpanBudget(max_chars)
        self._max_chars = max_chars
        self._overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document on its own structure.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order; empty for a blank document.
        """
        if not document.text.strip():
            return []

        spans = self._spans(document)
        chunks: list[Chunk] = []
        for start, end, title in spans:
            body = document.text[start:end]
            if not body.strip():
                continue
            metadata = {"section_title": title} if title else {}
            for piece_start, piece_end in self._subdivide(document, start, end):
                piece = document.text[piece_start:piece_end]
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk.create(
                        doc_id=document.id,
                        ordinal=len(chunks),
                        text=piece,
                        char_start=piece_start,
                        char_end=piece_end,
                        metadata=dict(metadata),
                    )
                )
        return chunks

    def _spans(self, document: Document) -> list[tuple[int, int, str]]:
        """Top-level unit spans, or paragraph groups when the document has no structure."""
        sections = document.sections_at_level(_TOP_LEVEL)
        if sections:
            return [(s.start, min(s.end, len(document.text)), s.title) for s in sections]
        return [
            (start, end, "") for start, end in _paragraph_groups(document.text, self._max_chars)
        ]

    def _subdivide(self, document: Document, start: int, end: int) -> list[tuple[int, int]]:
        """Break one over-long unit at its own subdivisions, then at paragraph breaks."""
        if self._budget.fits(start, end):
            return [(start, end)]

        inner = [
            s
            for s in document.sections_at_level(_SUB_LEVEL)
            if s.start >= start and s.end <= end and s.end > s.start
        ]
        if inner:
            spans = _merge_to_budget(
                [(s.start, s.end) for s in _fill_gaps(inner, start, end)], self._max_chars
            )
        else:
            groups = _paragraph_groups(document.text[start:end], self._max_chars, self._overlap)
            spans = [(start + s, start + e) for s, e in groups]
        return self._budget.enforce(spans)


def _fill_gaps(sections: list[DocumentSection], start: int, end: int) -> list[DocumentSection]:
    """Extend the first and last subdivision so no text between them is dropped."""
    ordered = sorted(sections, key=lambda s: s.start)
    first = ordered[0].model_copy(update={"start": start})
    last = ordered[-1].model_copy(update={"end": end})
    if len(ordered) == 1:
        return [first.model_copy(update={"end": end})]
    return [first, *ordered[1:-1], last]


def _merge_to_budget(spans: list[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    """Glue adjacent spans together while they fit, keeping related subdivisions intact."""
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and end - merged[-1][0] <= max_chars:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _paragraph_groups(text: str, max_chars: int, overlap: int = 0) -> list[tuple[int, int]]:
    """Group blank-line-separated paragraphs into spans no longer than the budget."""
    groups: list[tuple[int, int]] = []
    offset = 0
    current_start: int | None = None
    current_end = 0

    for paragraph in text.split(_PARAGRAPH_SEPARATOR):
        start = offset
        end = offset + len(paragraph)
        offset = end + len(_PARAGRAPH_SEPARATOR)
        if not paragraph.strip():
            continue
        if current_start is None:
            current_start, current_end = start, end
        elif end - current_start <= max_chars:
            current_end = end
        else:
            groups.append((current_start, current_end))
            current_start = max(start - overlap, current_end - overlap) if overlap else start
            current_end = end

    if current_start is not None:
        groups.append((current_start, current_end))
    return groups or [(0, len(text))]
