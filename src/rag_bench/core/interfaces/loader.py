"""Contract for corpus loaders, pipeline stage 1."""

from abc import ABC, abstractmethod
from pathlib import Path

from rag_bench.core.models import Document


class BaseLoader(ABC):
    """Turns a directory of raw source files into :class:`Document` objects.

    A loader owns everything format- and corpus-specific: parsing, cleaning, and marking
    up the section boundaries that make retrieval quality measurable.
    """

    @abstractmethod
    def load(self, root: Path) -> list[Document]:
        """Load every document under a corpus directory.

        Args:
            root: Directory holding the cached corpus files.

        Returns:
            The parsed documents, each with its sections populated.

        Raises:
            CorpusError: If the directory is missing or holds nothing loadable.
        """
