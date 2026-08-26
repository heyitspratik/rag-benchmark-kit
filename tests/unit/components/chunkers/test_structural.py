import pytest

from rag_bench.components.chunkers.structural import StructuralChunker
from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.models import Document, DocumentSection

from .conftest import make_document


def test_one_chunk_per_top_level_section(structured: Document) -> None:
    chunks = StructuralChunker(max_chars=1200, overlap=0).chunk(structured)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("Article 1")
    assert chunks[1].text.startswith("Article 2")


def test_section_titles_are_carried_into_metadata(structured: Document) -> None:
    chunks = StructuralChunker(max_chars=1200, overlap=0).chunk(structured)

    assert [c.metadata["section_title"] for c in chunks] == ["One", "Two"]


def test_offsets_match_the_chunk_text(structured: Document) -> None:
    chunks = StructuralChunker(max_chars=1200, overlap=0).chunk(structured)

    assert all(structured.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_an_over_long_section_is_split_at_its_own_subdivisions() -> None:
    first = "1. " + "a" * 300
    second = "2. " + "b" * 300
    text = f"Article 1\n\n{first}\n\n{second}"
    document = make_document(
        text,
        [
            DocumentSection(ref="Art. 1", start=0, end=len(text), level=1),
            DocumentSection(ref="Art. 1(1)", start=11, end=11 + len(first), level=2),
            DocumentSection(ref="Art. 1(2)", start=13 + len(first), end=len(text), level=2),
        ],
    )

    chunks = StructuralChunker(max_chars=320, overlap=0).chunk(document)

    # The cut lands between the two numbered paragraphs, and the heading stays attached
    # to the first rather than being orphaned into a chunk of its own.
    assert len(chunks) == 2
    assert chunks[0].text.startswith("Article 1")
    assert first in chunks[0].text
    assert chunks[1].text == second


def test_no_chunk_exceeds_the_budget_even_without_subdivisions() -> None:
    text = "Article 1\n\n" + "a" * 2000
    document = make_document(text, [DocumentSection(ref="Art. 1", start=0, end=len(text), level=1)])

    chunks = StructuralChunker(max_chars=250, overlap=0).chunk(document)

    assert max(len(c.text) for c in chunks) <= 250


def test_a_document_without_sections_falls_back_to_paragraphs(prose: Document) -> None:
    chunks = StructuralChunker(max_chars=200, overlap=0).chunk(prose)

    assert len(chunks) > 1
    assert max(len(c.text) for c in chunks) <= 200


def test_the_whole_document_is_covered(structured: Document) -> None:
    chunks = StructuralChunker(max_chars=1200, overlap=0).chunk(structured)

    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(structured.text)


def test_a_document_shorter_than_the_budget_yields_one_chunk() -> None:
    chunks = StructuralChunker(max_chars=1200, overlap=0).chunk(make_document("short text"))

    assert [c.text for c in chunks] == ["short text"]


@pytest.mark.parametrize("text", ["", "  \n\n "])
def test_an_empty_document_yields_nothing(text: str) -> None:
    assert StructuralChunker().chunk(make_document(text)) == []


@pytest.mark.parametrize(
    ("max_chars", "overlap", "message"),
    [
        (0, 0, "max_chars must be positive"),
        (100, -1, "must not be negative"),
        (100, 100, "smaller than max_chars"),
    ],
)
def test_invalid_parameters_are_rejected(max_chars: int, overlap: int, message: str) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        StructuralChunker(max_chars=max_chars, overlap=overlap)
