"""Reciprocal Rank Fusion, the arithmetic that combines two rankings into one.

Kept apart from the retrievers because it is pure arithmetic with no dependency on an
embedder or a store, which makes it both directly testable and reusable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_bench.core.exceptions import ConfigValidationError

#: The constant from Cormack, Clarke and Buettcher (2009). It damps the influence of the
#: very top ranks; smaller values let a single ranker's first result dominate the fusion.
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class RankedList:
    """One ranker's output: its name, how much it counts, and its ordering."""

    name: str
    weight: float
    ids: tuple[str, ...]


@dataclass(frozen=True)
class FusedResult:
    """One item's combined standing across every ranker."""

    id: str
    score: float
    best_rank: int


class ReciprocalRankFusion:
    """Combines rankings by position rather than by score.

    Scores from a vector search and from BM25 are on unrelated scales, so averaging them
    is meaningless. Ranks are comparable, which is why fusing on rank works where fusing
    on score does not.
    """

    def __init__(self, k: int = DEFAULT_RRF_K) -> None:
        """Initialise the fusion.

        Args:
            k: The damping constant.

        Raises:
            ConfigValidationError: If ``k`` is negative.
        """
        if k < 0:
            raise ConfigValidationError(f"retriever.params.rrf_k must not be negative, got {k}")
        self._k = k

    @property
    def k(self) -> int:
        """The damping constant in use."""
        return self._k

    def fuse(self, rankings: Sequence[RankedList]) -> list[FusedResult]:
        """Merge several rankings into one, best first.

        Args:
            rankings: One entry per ranker. Rankers contributing no results, and rankers
                weighted at zero, are ignored.

        Returns:
            Every item seen by any ranker, ordered by descending fused score. Ties break
            on best rank and then on ID, so the ordering is fully deterministic and a
            benchmark rerun cannot shuffle equally scored results.
        """
        scores: dict[str, float] = {}
        best_ranks: dict[str, int] = {}

        for ranking in rankings:
            if ranking.weight == 0.0:
                continue
            for rank, item_id in enumerate(ranking.ids):
                scores[item_id] = scores.get(item_id, 0.0) + ranking.weight / (self._k + rank + 1)
                best_ranks[item_id] = min(best_ranks.get(item_id, rank), rank)

        return [
            FusedResult(id=item_id, score=scores[item_id], best_rank=best_ranks[item_id])
            for item_id in sorted(scores, key=lambda i: (-scores[i], best_ranks[i], i))
        ]
