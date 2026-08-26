"""Abstract base classes defining the contract for every swappable pipeline stage."""

from rag_bench.core.interfaces.chunker import BaseChunker
from rag_bench.core.interfaces.embedder import BaseEmbedder, Vector
from rag_bench.core.interfaces.generator import BaseGenerator
from rag_bench.core.interfaces.loader import BaseLoader
from rag_bench.core.interfaces.retriever import BaseRetriever
from rag_bench.core.interfaces.store import BaseVectorStore

__all__ = [
    "BaseChunker",
    "BaseEmbedder",
    "BaseGenerator",
    "BaseLoader",
    "BaseRetriever",
    "BaseVectorStore",
    "Vector",
]
