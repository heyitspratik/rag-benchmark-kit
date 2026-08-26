"""Contract for answer generators, pipeline stage 6."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag_bench.core.models import Answer, ScoredChunk


class BaseGenerator(ABC):
    """Turns a question plus retrieved context into a cited answer."""

    @abstractmethod
    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        """Answer a question from the supplied context only.

        Args:
            question: The user's question.
            contexts: Retrieved chunks, best first.

        Returns:
            The answer, with every citation resolved to a chunk that was actually
            supplied, and ``abstained`` set when the context was insufficient.

        Raises:
            CitationError: If the model cited context that was not supplied.
            LLMProviderError: If the provider call failed.
        """
