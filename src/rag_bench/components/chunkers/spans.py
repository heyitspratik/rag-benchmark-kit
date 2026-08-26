"""The character budget shared by the structure-aware chunkers.

Both the structural and the semantic strategy cut on a signal unrelated to length: an
article boundary, a topic shift. Either can therefore produce a unit far longer than the
configured budget, and an over-long chunk quietly wrecks a benchmark by blowing past the
generator's context limit. Applying the budget as a final pass leaves those strategies
free to cut where they believe the meaning changes.
"""

from __future__ import annotations

from rag_bench.core.exceptions import ConfigValidationError

type Span = tuple[int, int]


class SpanBudget:
    """A maximum chunk length, and the cuts needed to keep spans inside it."""

    def __init__(self, max_chars: int) -> None:
        """Initialise the budget.

        Args:
            max_chars: Longest span to allow.

        Raises:
            ConfigValidationError: If the budget is not positive.
        """
        if max_chars <= 0:
            raise ConfigValidationError(
                f"chunker.params.max_chars must be positive, got {max_chars}"
            )
        self._max_chars = max_chars

    @property
    def max_chars(self) -> int:
        """The configured maximum span length."""
        return self._max_chars

    def fits(self, start: int, end: int) -> bool:
        """Whether a span is already within budget.

        Args:
            start: Inclusive start offset.
            end: Exclusive end offset.

        Returns:
            True when the span needs no splitting.
        """
        return end - start <= self._max_chars

    def enforce(self, spans: list[Span]) -> list[Span]:
        """Split any span over budget, leaving the rest untouched.

        Args:
            spans: Character spans in document order.

        Returns:
            Spans in document order, none longer than the budget.
        """
        capped: list[Span] = []
        for start, end in spans:
            if self.fits(start, end):
                capped.append((start, end))
            else:
                capped.extend(self.hard_split(start, end))
        return capped

    def hard_split(self, start: int, end: int) -> list[Span]:
        """Cut a span into equal windows, ignoring content entirely.

        The last resort, used only where no structural or semantic boundary is available.

        Args:
            start: Inclusive start offset.
            end: Exclusive end offset.

        Returns:
            Consecutive, non-overlapping windows covering the span.
        """
        return [
            (cut, min(cut + self._max_chars, end)) for cut in range(start, end, self._max_chars)
        ]
