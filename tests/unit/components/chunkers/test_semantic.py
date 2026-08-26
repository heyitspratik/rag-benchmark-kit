import pytest

from rag_bench.components.chunkers.semantic import SemanticChunker
from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.models import Document

from .conftest import FAKE_EMBEDDER_NAME, make_document


def _chunker(**overrides: object) -> SemanticChunker:
    params: dict[str, object] = {
        "embedder": FAKE_EMBEDDER_NAME,
        "max_chars": 1200,
        "min_chunk_chars": 0,
        "breakpoint_percentile": 50.0,
    }
    return SemanticChunker(**(params | overrides))  # type: ignore[arg-type]


def _topic_document() -> Document:
    alpha = " ".join(["alpha talks about the first topic."] * 3)
    beta = " ".join(["beta covers an entirely different topic."] * 3)
    return make_document(f"{alpha} {beta}")


def test_the_cut_lands_where_the_topic_changes() -> None:
    document = _topic_document()

    chunks = _chunker().chunk(document)

    assert len(chunks) == 2
    assert "beta" not in chunks[0].text
    assert "alpha" not in chunks[1].text


def test_offsets_match_the_chunk_text() -> None:
    document = _topic_document()

    chunks = _chunker().chunk(document)

    assert all(document.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_the_length_cap_is_a_real_bound_not_one_sentence_of_slack() -> None:
    # Every sentence is identical, so no topic breakpoint ever fires and only the cap
    # can end a chunk.
    document = make_document(" ".join(["alpha word word word word word."] * 60))

    chunks = _chunker(max_chars=120).chunk(document)

    assert max(len(c.text) for c in chunks) <= 120


def test_a_sentence_longer_than_the_cap_is_still_split() -> None:
    document = make_document("alpha " + "x" * 500)

    chunks = _chunker(max_chars=100).chunk(document)

    assert max(len(c.text) for c in chunks) <= 100


def test_short_chunks_are_not_cut_below_the_minimum() -> None:
    document = _topic_document()

    chunks = _chunker(min_chunk_chars=10_000).chunk(document)

    assert len(chunks) == 1


def test_a_document_with_too_few_sentences_is_one_chunk() -> None:
    document = make_document("alpha only sentence here.")

    chunks = _chunker().chunk(document)

    assert [c.text for c in chunks] == ["alpha only sentence here."]


def test_a_document_with_no_split_points_still_chunks() -> None:
    document = make_document("alpha " + "y" * 400)

    chunks = _chunker(max_chars=150).chunk(document)

    assert len(chunks) >= 3


@pytest.mark.parametrize("text", ["", "  \n\n "])
def test_an_empty_document_yields_nothing(text: str) -> None:
    assert _chunker().chunk(make_document(text)) == []


def test_the_embedder_is_resolved_lazily() -> None:
    # Constructing must not touch the registry, so a chunker can be built before the
    # embedder modules have been imported.
    chunker = SemanticChunker(embedder="does_not_exist")

    assert chunker is not None


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"max_chars": 0}, "max_chars must be positive"),
        ({"breakpoint_percentile": 0.0}, "between 0 and 100"),
        ({"breakpoint_percentile": 100.0}, "between 0 and 100"),
        ({"min_chunk_chars": -1}, "must not be negative"),
    ],
)
def test_invalid_parameters_are_rejected(params: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        _chunker(**params)
