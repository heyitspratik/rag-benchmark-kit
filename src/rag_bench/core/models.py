"""The domain objects that flow between pipeline stages.

These types are the contract the whole pipeline shares: a loader produces
:class:`Document`, a chunker turns it into :class:`Chunk` objects, a retriever returns
:class:`ScoredChunk` objects, and a generator produces an :class:`Answer`.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

# A fixed namespace makes chunk IDs deterministic across machines and re-runs, so
# rebuilding an index is idempotent and benchmark results stay comparable between runs.
CHUNK_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class DocumentSection(BaseModel):
    """A named, addressable span of a document, such as a single GDPR article.

    Sections are what make retrieval quality comparable across chunking strategies: the
    ground truth is expressed in section references, and any chunk can be mapped back to
    the sections it overlaps through its character offsets.
    """

    model_config = ConfigDict(frozen=True)

    ref: str = Field(description="Citable reference, e.g. 'GDPR Art. 15(4)'.")
    title: str = ""
    start: NonNegativeInt = Field(description="Inclusive character offset into the document.")
    end: NonNegativeInt = Field(description="Exclusive character offset into the document.")


class Document(BaseModel):
    """A whole source document as loaded from the corpus, before chunking."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    source: str = Field(description="URL or path the document was loaded from.")
    text: str
    sections: tuple[DocumentSection, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)

    def sections_overlapping(self, start: int, end: int) -> tuple[str, ...]:
        """Return the refs of every section overlapping the given character span.

        Args:
            start: Inclusive start offset.
            end: Exclusive end offset.

        Returns:
            The matching section refs, in document order.
        """
        return tuple(s.ref for s in self.sections if s.start < end and start < s.end)


class Chunk(BaseModel):
    """A retrievable unit of text produced by a chunker."""

    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    ordinal: NonNegativeInt = Field(description="Zero-based position within its document.")
    text: str
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    section_refs: tuple[str, ...] = Field(
        default=(),
        description="Sections this chunk overlaps; filled by the indexer, not the chunker.",
    )
    metadata: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        doc_id: str,
        ordinal: int,
        text: str,
        char_start: int,
        char_end: int,
        metadata: dict[str, str] | None = None,
    ) -> Chunk:
        """Build a chunk with a deterministic ID derived from its document and content.

        Args:
            doc_id: Identifier of the document this chunk came from.
            ordinal: Zero-based position within the document.
            text: The chunk body.
            char_start: Inclusive offset of the chunk in the source document.
            char_end: Exclusive offset of the chunk in the source document.
            metadata: Optional per-chunk metadata, such as an article heading.

        Returns:
            The constructed chunk.
        """
        chunk_id = uuid.uuid5(CHUNK_ID_NAMESPACE, f"{doc_id}:{ordinal}:{text}")
        return cls(
            id=str(chunk_id),
            doc_id=doc_id,
            ordinal=ordinal,
            text=text,
            char_start=char_start,
            char_end=char_end,
            metadata=metadata or {},
        )

    def with_sections(self, refs: tuple[str, ...]) -> Chunk:
        """Return a copy of this chunk annotated with the section refs it overlaps."""
        return self.model_copy(update={"section_refs": refs})


class ScoredChunk(BaseModel):
    """A chunk together with the score and rank a retriever assigned to it."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    rank: NonNegativeInt = Field(description="Zero-based position in the result list.")


class Citation(BaseModel):
    """A citation marker emitted by the generator, resolved to a real chunk."""

    model_config = ConfigDict(frozen=True)

    marker: str = Field(description="The label as the model wrote it, e.g. '3'.")
    chunk_id: str
    section_refs: tuple[str, ...] = ()


class TokenUsage(BaseModel):
    """Token counts reported by the LLM provider for a single generation."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: NonNegativeInt = 0
    completion_tokens: NonNegativeInt = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by the call."""
        return self.prompt_tokens + self.completion_tokens


class Answer(BaseModel):
    """The end-to-end result of answering one question."""

    model_config = ConfigDict(frozen=True)

    question: str
    text: str
    abstained: bool = Field(
        description="True when the model declined because the context was insufficient.",
    )
    citations: tuple[Citation, ...] = ()
    contexts: tuple[ScoredChunk, ...] = ()
    usage: TokenUsage = TokenUsage()
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    @property
    def cited_section_refs(self) -> tuple[str, ...]:
        """Every distinct section ref backing this answer, in citation order."""
        seen: dict[str, None] = {}
        for citation in self.citations:
            for ref in citation.section_refs:
                seen.setdefault(ref, None)
        return tuple(seen)
