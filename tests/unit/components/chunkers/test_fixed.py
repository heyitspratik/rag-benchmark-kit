import pytest

from rag_bench.components.chunkers.fixed import FixedChunker
from rag_bench.core.exceptions import ConfigValidationError

from .conftest import make_document


def test_windows_advance_by_the_stride() -> None:
    document = make_document("abcdefghij" * 10)

    chunks = FixedChunker(max_chars=40, overlap=10).chunk(document)

    assert [(c.char_start, c.char_end) for c in chunks][:3] == [(0, 40), (30, 70), (60, 100)]


def test_offsets_match_the_chunk_text() -> None:
    document = make_document("abcdefghij" * 10)

    chunks = FixedChunker(max_chars=40, overlap=10).chunk(document)

    assert all(document.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_no_chunk_exceeds_the_window() -> None:
    document = make_document("x" * 1000)

    chunks = FixedChunker(max_chars=120, overlap=20).chunk(document)

    assert max(len(c.text) for c in chunks) == 120


def test_ordinals_are_contiguous_from_zero() -> None:
    document = make_document("x" * 500)

    chunks = FixedChunker(max_chars=100, overlap=10).chunk(document)

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_a_document_shorter_than_the_window_yields_one_chunk() -> None:
    document = make_document("short text")

    chunks = FixedChunker(max_chars=1200, overlap=150).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert (chunks[0].char_start, chunks[0].char_end) == (0, 10)


def test_a_document_with_no_split_points_still_chunks() -> None:
    document = make_document("a" * 250)

    chunks = FixedChunker(max_chars=100, overlap=0).chunk(document)

    assert [len(c.text) for c in chunks] == [100, 100, 50]


@pytest.mark.parametrize("text", ["", "   \n\n  \t "])
def test_an_empty_document_yields_nothing(text: str) -> None:
    assert FixedChunker().chunk(make_document(text)) == []


def test_zero_overlap_is_allowed() -> None:
    chunks = FixedChunker(max_chars=10, overlap=0).chunk(make_document("abcdefghijklm"))

    assert [c.text for c in chunks] == ["abcdefghij", "klm"]


@pytest.mark.parametrize(
    ("max_chars", "overlap", "message"),
    [
        (0, 0, "max_chars must be positive"),
        (-5, 0, "max_chars must be positive"),
        (100, -1, "overlap must not be negative"),
        (100, 100, "never advances"),
        (100, 150, "never advances"),
    ],
)
def test_parameters_that_cannot_make_progress_are_rejected(
    max_chars: int, overlap: int, message: str
) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        FixedChunker(max_chars=max_chars, overlap=overlap)
