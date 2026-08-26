import pytest

from rag_bench.core.models import (
    Answer,
    Chunk,
    Citation,
    Document,
    DocumentSection,
    ScoredChunk,
    TokenUsage,
)


def _document() -> Document:
    return Document(
        id="gdpr",
        title="GDPR",
        source="https://example.invalid/gdpr",
        text="a" * 100,
        sections=(
            DocumentSection(ref="GDPR Art. 1", start=0, end=40),
            DocumentSection(ref="GDPR Art. 2", start=40, end=100),
        ),
    )


def test_chunk_id_is_deterministic_for_identical_input() -> None:
    kwargs = {"doc_id": "gdpr", "ordinal": 3, "text": "hello", "char_start": 0, "char_end": 5}

    assert Chunk.create(**kwargs).id == Chunk.create(**kwargs).id


def test_chunk_id_changes_with_text_and_position() -> None:
    base = Chunk.create(doc_id="gdpr", ordinal=0, text="hello", char_start=0, char_end=5)
    other_text = Chunk.create(doc_id="gdpr", ordinal=0, text="world", char_start=0, char_end=5)
    other_ordinal = Chunk.create(doc_id="gdpr", ordinal=1, text="hello", char_start=0, char_end=5)

    assert len({base.id, other_text.id, other_ordinal.id}) == 3


def test_chunks_are_immutable() -> None:
    chunk = Chunk.create(doc_id="d", ordinal=0, text="t", char_start=0, char_end=1)

    with pytest.raises(ValueError, match="frozen"):
        chunk.text = "changed"  # type: ignore[misc]


def test_sections_overlapping_covers_partial_spans() -> None:
    doc = _document()

    assert doc.sections_overlapping(0, 10) == ("GDPR Art. 1",)
    assert doc.sections_overlapping(30, 50) == ("GDPR Art. 1", "GDPR Art. 2")
    assert doc.sections_overlapping(40, 41) == ("GDPR Art. 2",)


def test_sections_overlapping_excludes_touching_boundaries() -> None:
    doc = _document()

    # A chunk ending exactly where a section starts does not overlap it.
    assert doc.sections_overlapping(0, 40) == ("GDPR Art. 1",)


def test_with_sections_returns_annotated_copy() -> None:
    chunk = Chunk.create(doc_id="d", ordinal=0, text="t", char_start=0, char_end=1)

    annotated = chunk.with_sections(("GDPR Art. 5",))

    assert chunk.section_refs == ()
    assert annotated.section_refs == ("GDPR Art. 5",)
    assert annotated.id == chunk.id


def test_token_usage_totals() -> None:
    assert TokenUsage(prompt_tokens=10, completion_tokens=5).total_tokens == 15


def test_cited_section_refs_deduplicates_preserving_order() -> None:
    answer = Answer(
        question="q",
        text="a",
        abstained=False,
        citations=(
            Citation(marker="1", chunk_id="c1", section_refs=("Art. 5", "Art. 6")),
            Citation(marker="2", chunk_id="c2", section_refs=("Art. 6", "Art. 7")),
        ),
    )

    assert answer.cited_section_refs == ("Art. 5", "Art. 6", "Art. 7")


def test_scored_chunk_carries_rank_and_score() -> None:
    chunk = Chunk.create(doc_id="d", ordinal=0, text="t", char_start=0, char_end=1)

    scored = ScoredChunk(chunk=chunk, score=0.87, rank=0)

    assert (scored.rank, scored.score) == (0, 0.87)
