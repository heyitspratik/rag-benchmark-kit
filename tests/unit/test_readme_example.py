"""The README's 'Extending' worked example, run verbatim.

That section is what converts a reader into a user, so it is executed rather than
trusted. If the registry API or the Chunk constructor changes, this fails and the README
gets corrected instead of quietly becoming wrong.
"""

from pathlib import Path

from rag_bench.core.interfaces import BaseChunker
from rag_bench.core.models import Chunk, Document
from rag_bench.core.registry import CHUNKERS, register_chunker

EXAMPLE_NAME = "sentence"


@register_chunker(EXAMPLE_NAME)
class SentenceChunker(BaseChunker):
    """One chunk per sentence, skipping fragments."""

    def __init__(self, min_chars: int = 40) -> None:
        self._min_chars = min_chars

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        for piece in document.text.split(". "):
            end = start + len(piece)
            if len(piece.strip()) >= self._min_chars:
                chunks.append(
                    Chunk.create(
                        doc_id=document.id,
                        ordinal=len(chunks),
                        text=piece,
                        char_start=start,
                        char_end=end,
                    )
                )
            start = end + 2
        return chunks


_TEXT = (
    "The controller shall provide information without undue delay. "
    "That period may be extended by two further months where necessary. Short."
)


def _document() -> Document:
    return Document(id="d", title="t", source="s", text=_TEXT)


def test_a_decorated_class_is_resolvable_by_name() -> None:
    # This is the whole claim of the extension point: a class and a decorator, and the
    # config layer can reach it.
    built = CHUNKERS.create("sentence", {"min_chars": 40})

    assert isinstance(built, SentenceChunker)


def test_the_example_chunks_as_the_readme_says() -> None:
    chunks = CHUNKERS.create("sentence", {"min_chars": 40}).chunk(_document())

    assert len(chunks) == 2
    assert chunks[0].text.startswith("The controller")
    assert "Short." not in [c.text for c in chunks]


def test_the_example_sets_truthful_offsets() -> None:
    # The README calls this "the one rule", because these offsets are what make a new
    # strategy comparable to the existing ones.
    document = _document()

    chunks = CHUNKERS.create("sentence").chunk(document)

    assert all(document.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_the_readme_still_contains_this_example() -> None:
    # Guards the other direction: if the section is renamed or removed, this test says so.
    readme = Path("README.md").read_text()

    assert '@register_chunker("sentence")' in readme
    assert "class SentenceChunker(BaseChunker):" in readme


def test_every_component_the_readme_advertises_is_registered() -> None:
    from rag_bench.components import load_components
    from rag_bench.core.registry import available_components

    load_components()
    registered = available_components()
    table = Path("README.md").read_text()
    section = table.split("| Stage | Available |", 1)[1].split("\n\n", 1)[0]

    for stage in ("chunker", "embedder", "store", "retriever", "generator"):
        for name in registered[stage]:
            # Test doubles and the README's own worked example are registered by the
            # suite itself and are deliberately not shipped components.
            if name == EXAMPLE_NAME or name.startswith("_test"):
                continue
            assert f"`{name}`" in section, f"{stage} {name!r} is registered but undocumented"
