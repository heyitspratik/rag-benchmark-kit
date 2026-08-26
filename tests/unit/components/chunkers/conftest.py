"""Shared document builders and a deterministic stand-in embedder."""

from collections.abc import Sequence

import pytest

from rag_bench.core.interfaces import BaseEmbedder, Vector
from rag_bench.core.models import Document, DocumentSection
from rag_bench.core.registry import register_embedder

#: Registered once, under a name no real config would use, so the semantic chunker can be
#: exercised through the registry it actually resolves through.
FAKE_EMBEDDER_NAME = "_test_topic_embedder"


@register_embedder(FAKE_EMBEDDER_NAME)
class TopicEmbedder(BaseEmbedder):
    """Embeds by the first word of a sentence, so topic shifts are exactly predictable."""

    def __init__(self, topics: Sequence[str] = ("alpha", "beta")) -> None:
        self._topics = list(topics)

    @property
    def dimension(self) -> int:
        return len(self._topics)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)

    def _vector(self, text: str) -> Vector:
        first = text.strip().split(" ")[0].lower().strip(".,")
        return [1.0 if topic == first else 0.0 for topic in self._topics]


def make_document(text: str, sections: Sequence[DocumentSection] = ()) -> Document:
    """Build a document for chunker tests.

    Args:
        text: The document body.
        sections: Structural sections, if the strategy under test needs them.

    Returns:
        The document.
    """
    return Document(id="doc", title="Doc", source="test", text=text, sections=tuple(sections))


@pytest.fixture
def prose() -> Document:
    """A document with paragraph breaks but no declared structure."""
    paragraph = "Sentence one is here. Sentence two follows on. Sentence three ends it."
    return make_document("\n\n".join([paragraph] * 12))


@pytest.fixture
def structured() -> Document:
    """A document whose sections nest paragraphs inside articles."""
    article_one = "Article 1\n\n1. First rule text.\n\n2. Second rule text."
    article_two = "Article 2\n\n1. Third rule text."
    text = f"{article_one}\n\n{article_two}"
    split = len(article_one) + 2
    return make_document(
        text,
        [
            DocumentSection(ref="Art. 1", title="One", start=0, end=len(article_one), level=1),
            DocumentSection(ref="Art. 1(1)", start=11, end=32, level=2),
            DocumentSection(ref="Art. 1(2)", start=34, end=len(article_one), level=2),
            DocumentSection(ref="Art. 2", title="Two", start=split, end=len(text), level=1),
            DocumentSection(ref="Art. 2(1)", start=split + 11, end=len(text), level=2),
        ],
    )
