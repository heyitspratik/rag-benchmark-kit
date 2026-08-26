"""Contract for embedding models, pipeline stage 3."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

type Vector = list[float]


class BaseEmbedder(ABC):
    """Turns text into vectors.

    Documents and queries are embedded through separate methods on purpose: several
    models (E5, BGE) expect asymmetric prefixes or instructions, and that asymmetry
    belongs inside the implementation rather than at every call site.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors this embedder produces."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch of passages.

        Args:
            texts: Passage texts, in the order the vectors should be returned.

        Returns:
            One vector per input text.
        """

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """Embed a single search query.

        Args:
            text: The user's question.

        Returns:
            The query vector.
        """
