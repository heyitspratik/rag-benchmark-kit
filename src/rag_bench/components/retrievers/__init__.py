"""Retrieval strategies. Importing this package registers every implementation."""

from rag_bench.components.retrievers.dense import DenseRetriever
from rag_bench.components.retrievers.fusion import (
    FusedResult,
    RankedList,
    ReciprocalRankFusion,
)
from rag_bench.components.retrievers.hybrid import HybridRetriever
from rag_bench.components.retrievers.hybrid_rerank import HybridRerankRetriever

__all__ = [
    "DenseRetriever",
    "FusedResult",
    "HybridRerankRetriever",
    "HybridRetriever",
    "RankedList",
    "ReciprocalRankFusion",
]
