import pytest

from rag_bench.components.chunkers.recursive import RecursiveChunker
from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.models import Document

from .conftest import make_document


def test_offsets_match_the_chunk_text(prose: Document) -> None:
    chunks = RecursiveChunker(max_chars=200, overlap=20).chunk(prose)

    assert all(prose.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_chunks_stay_within_budget(prose: Document) -> None:
    chunks = RecursiveChunker(max_chars=200, overlap=20).chunk(prose)

    assert max(len(c.text) for c in chunks) <= 200


def test_paragraph_boundaries_are_preferred_over_mid_sentence_cuts() -> None:
    document = make_document("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")

    chunks = RecursiveChunker(max_chars=25, overlap=0).chunk(document)

    assert [c.text.strip() for c in chunks] == [
        "First paragraph.",
        "Second paragraph.",
        "Third paragraph.",
    ]


def test_repeated_text_gets_distinct_offsets() -> None:
    # A naive implementation would locate chunks with str.find and collapse duplicates
    # onto the first occurrence.
    document = make_document("\n\n".join(["Identical paragraph."] * 4))

    chunks = RecursiveChunker(max_chars=25, overlap=0).chunk(document)

    assert len({c.char_start for c in chunks}) == len(chunks)


def test_a_document_shorter_than_the_budget_yields_one_chunk() -> None:
    chunks = RecursiveChunker(max_chars=1200, overlap=150).chunk(make_document("short text"))

    assert [c.text for c in chunks] == ["short text"]


def test_a_document_with_no_split_points_still_chunks() -> None:
    chunks = RecursiveChunker(max_chars=50, overlap=0).chunk(make_document("a" * 160))

    assert len(chunks) >= 3
    assert max(len(c.text) for c in chunks) <= 50


@pytest.mark.parametrize("text", ["", "   \n\n  "])
def test_an_empty_document_yields_nothing(text: str) -> None:
    assert RecursiveChunker().chunk(make_document(text)) == []


@pytest.mark.parametrize(
    ("max_chars", "overlap", "message"),
    [(0, 0, "max_chars must be positive"), (100, 100, "smaller than max_chars")],
)
def test_invalid_parameters_are_rejected(max_chars: int, overlap: int, message: str) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        RecursiveChunker(max_chars=max_chars, overlap=overlap)
