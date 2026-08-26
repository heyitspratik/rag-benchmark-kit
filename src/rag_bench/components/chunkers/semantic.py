"""Semantic chunking: cut where the topic changes rather than where the budget runs out."""

from __future__ import annotations

import re

import numpy as np

from rag_bench.components.chunkers.spans import SpanBudget
from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.interfaces import BaseChunker, BaseEmbedder
from rag_bench.core.models import Chunk, Document
from rag_bench.core.registry import EMBEDDERS, register_chunker

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|\n\n+")

_MIN_SENTENCES_FOR_ANALYSIS = 3


@register_chunker("semantic")
class SemanticChunker(BaseChunker):
    """Starts a new chunk where consecutive sentences stop resembling each other.

    Every adjacent pair of sentences is scored by cosine similarity, and the lowest
    ``breakpoint_percentile`` of those scores become cut points. The threshold is a
    percentile of the document's own distribution rather than an absolute value, because
    what counts as a topic shift differs between a regulation and a tutorial. A document
    whose sentences are uniformly similar yields no breakpoints at all, and is cut by the
    length cap alone.

    This is the most expensive strategy: it embeds every sentence before it can chunk
    anything, which the benchmark should show is not always repaid in answer quality.
    """

    def __init__(
        self,
        max_chars: int = 1200,
        breakpoint_percentile: float = 25.0,
        embedder: str = "bge_small",
        embedder_params: dict[str, object] | None = None,
        min_chunk_chars: int = 200,
    ) -> None:
        """Initialise the chunker.

        Args:
            max_chars: Hard cap; a run of similar sentences is still cut at this length.
            breakpoint_percentile: Percentile of similarity scores treated as breaks.
            embedder: Registered embedder name used to score sentence similarity.
            embedder_params: Constructor parameters for that embedder.
            min_chunk_chars: Chunks shorter than this are merged into the next one.

        Raises:
            ConfigValidationError: If any bound is outside its valid range.
        """
        if max_chars <= 0:
            raise ConfigValidationError(
                f"chunker.params.max_chars must be positive, got {max_chars}"
            )
        if not 0.0 < breakpoint_percentile < 100.0:
            raise ConfigValidationError(
                "chunker.params.breakpoint_percentile must be between 0 and 100 exclusive, "
                f"got {breakpoint_percentile}"
            )
        if min_chunk_chars < 0:
            raise ConfigValidationError(
                f"chunker.params.min_chunk_chars must not be negative, got {min_chunk_chars}"
            )
        self._budget = SpanBudget(max_chars)
        self._max_chars = max_chars
        self._percentile = breakpoint_percentile
        self._min_chunk_chars = min_chunk_chars
        self._embedder_name = embedder
        self._embedder_params = embedder_params or {}
        self._embedder: BaseEmbedder | None = None

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document at its own topic boundaries.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order; empty for a blank document.
        """
        if not document.text.strip():
            return []

        sentences = _split_sentences(document.text)
        if len(sentences) < _MIN_SENTENCES_FOR_ANALYSIS:
            # Too little text for the similarity distribution to mean anything, but the
            # length cap still has to hold.
            grouped = [(sentences[0][0], sentences[-1][1])]
        else:
            breakpoints = self._breakpoints([document.text[s:e] for s, e in sentences])
            grouped = _group_sentences(
                sentences, breakpoints, self._max_chars, self._min_chunk_chars
            )

        spans = self._budget.enforce(grouped)
        return [
            Chunk.create(
                doc_id=document.id,
                ordinal=ordinal,
                text=document.text[start:end],
                char_start=start,
                char_end=end,
            )
            for ordinal, (start, end) in enumerate(spans)
            if document.text[start:end].strip()
        ]

    def _breakpoints(self, sentences: list[str]) -> set[int]:
        """Indexes after which the topic shifts enough to warrant a new chunk."""
        vectors = np.asarray(self._resolve_embedder().embed_documents(sentences), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit = vectors / np.where(norms == 0.0, 1.0, norms)
        similarities = np.sum(unit[:-1] * unit[1:], axis=1)
        threshold = float(np.percentile(similarities, self._percentile))
        # Strictly below, not at or below: when every sentence resembles its neighbour
        # equally the percentile equals them all, and a non-strict test would declare a
        # breakpoint after every sentence.
        return {int(i) for i in np.flatnonzero(similarities < threshold)}

    def _resolve_embedder(self) -> BaseEmbedder:
        """Build the scoring embedder on first use.

        Deferred rather than built in ``__init__`` so that constructing this chunker never
        depends on embedder modules having been imported first.
        """
        if self._embedder is None:
            self._embedder = EMBEDDERS.create(self._embedder_name, self._embedder_params)
        return self._embedder


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans, as offsets into the original text."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END_RE.finditer(text):
        if text[cursor : match.start()].strip():
            spans.append((cursor, match.start()))
        cursor = match.end()
    if text[cursor:].strip():
        spans.append((cursor, len(text)))
    return spans or [(0, len(text))]


def _group_sentences(
    sentences: list[tuple[int, int]],
    breakpoints: set[int],
    max_chars: int,
    min_chunk_chars: int,
) -> list[tuple[int, int]]:
    """Collect sentences into spans, cutting at breakpoints and at the length cap."""
    spans: list[tuple[int, int]] = []
    start, previous_end = sentences[0]

    for index, (sentence_start, sentence_end) in enumerate(sentences):
        # Close the chunk before a sentence that would push it over budget, rather than
        # after, so the cap is a real bound instead of one sentence of slack.
        if sentence_start > start and sentence_end - start > max_chars:
            spans.append((start, previous_end))
            start = sentence_start
        previous_end = sentence_end

        if index + 1 == len(sentences):
            spans.append((start, sentence_end))
        elif index in breakpoints and sentence_end - start >= min_chunk_chars:
            spans.append((start, sentence_end))
            start = sentences[index + 1][0]
            previous_end = start
    return spans
