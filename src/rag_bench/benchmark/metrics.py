"""Scoring a configuration's answers.

Most of what matters here is computed directly rather than judged by a model: hit rate,
reciprocal rank, abstention accuracy, latency and cost are deterministic, cost nothing,
and rerun identically. The RAGAS metrics are LLM-judged, carry their own variance, and
are optional for exactly that reason. A run without them still produces a usable table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median

from rag_bench.benchmark.evalset import Difficulty, EvalQuestion
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Answer

logger = get_logger(__name__)

_P95 = 0.95


@dataclass(frozen=True)
class QuestionOutcome:
    """One answered question, paired with the question it answers.

    Carrying both together is what lets every metric be computed from a single list
    without re-joining answers to questions by ID.
    """

    question: EvalQuestion
    answer: Answer
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def retrieved_refs(self) -> tuple[str, ...]:
        """Every section ref covered by the retrieved chunks, in rank order."""
        return tuple(ref for context in self.answer.contexts for ref in context.chunk.section_refs)

    @property
    def is_hit(self) -> bool:
        """Whether any retrieved chunk covers a section the ground truth cites."""
        return bool(set(self.question.source_refs) & set(self.retrieved_refs))

    @property
    def first_relevant_rank(self) -> int | None:
        """Zero-based rank of the first chunk covering a cited section, if any."""
        wanted = set(self.question.source_refs)
        for context in self.answer.contexts:
            if wanted & set(context.chunk.section_refs):
                return context.rank
        return None


@dataclass(frozen=True)
class MetricSet:
    """Aggregated scores for one configuration."""

    question_count: int
    hit_rate: float | None = None
    mrr: float | None = None
    abstention_accuracy: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    p50_retrieval_ms: float | None = None
    p95_retrieval_ms: float | None = None
    p50_generation_ms: float | None = None
    p95_generation_ms: float | None = None
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def as_dict(self) -> dict[str, float | int | None]:
        """The metrics as a plain mapping, for persistence and reporting."""
        return {
            "question_count": self.question_count,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "abstention_accuracy": self.abstention_accuracy,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "p50_retrieval_ms": self.p50_retrieval_ms,
            "p95_retrieval_ms": self.p95_retrieval_ms,
            "p50_generation_ms": self.p50_generation_ms,
            "p95_generation_ms": self.p95_generation_ms,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


class PriceTable:
    """Per-model token prices, used to turn token counts into a cost estimate."""

    def __init__(self, prices: dict[str, dict[str, float]] | None = None) -> None:
        """Initialise the table.

        Args:
            prices: Model name mapped to ``input_usd_per_1k`` and ``output_usd_per_1k``.
                Anything absent is treated as free, which is correct for a local model.
        """
        self._prices = prices or {}

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the cost of one generation.

        Args:
            model: The model name the run used.
            prompt_tokens: Input tokens consumed.
            completion_tokens: Output tokens produced.

        Returns:
            Cost in US dollars. Zero for a model with no listed price, since the default
            provider runs locally and genuinely costs nothing.
        """
        entry = self._prices.get(model)
        if entry is None:
            return 0.0
        return prompt_tokens / 1000 * entry.get(
            "input_usd_per_1k", 0.0
        ) + completion_tokens / 1000 * entry.get("output_usd_per_1k", 0.0)


def hit_rate(outcomes: Sequence[QuestionOutcome]) -> float | None:
    """Fraction of answerable questions where retrieval found a cited section.

    Args:
        outcomes: Scored outcomes, including negatives.

    Returns:
        The fraction, or ``None`` when there is nothing answerable to measure. Negative
        questions are excluded: they have no correct section, so counting them would
        drag the score down for behaving correctly.
    """
    answerable = [o for o in outcomes if not o.question.is_negative]
    if not answerable:
        return None
    return sum(1 for o in answerable if o.is_hit) / len(answerable)


def mean_reciprocal_rank(outcomes: Sequence[QuestionOutcome]) -> float | None:
    """Mean of ``1 / (rank + 1)`` for the first relevant chunk of each question.

    Args:
        outcomes: Scored outcomes, including negatives.

    Returns:
        The mean, counting a miss as zero, or ``None`` when nothing is answerable.
    """
    answerable = [o for o in outcomes if not o.question.is_negative]
    if not answerable:
        return None
    total = 0.0
    for outcome in answerable:
        rank = outcome.first_relevant_rank
        if rank is not None:
            total += 1.0 / (rank + 1)
    return total / len(answerable)


def abstention_accuracy(outcomes: Sequence[QuestionOutcome]) -> float | None:
    """Fraction of negative questions the system correctly declined to answer.

    Args:
        outcomes: Scored outcomes, including answerable ones.

    Returns:
        The fraction, or ``None`` when the set has no negative questions. This is the
        measurement that catches a configuration which looks strong only because it
        answers confidently even when it has nothing to go on.
    """
    negatives = [o for o in outcomes if o.question.is_negative]
    if not negatives:
        return None
    return sum(1 for o in negatives if o.answer.abstained) / len(negatives)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile of a sample.

    Args:
        values: The sample; may be unsorted.
        fraction: Percentile as a fraction, so 0.95 for p95.

    Returns:
        The value at that percentile, or ``None`` for an empty sample.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def aggregate(
    outcomes: Sequence[QuestionOutcome],
    *,
    model: str = "",
    prices: PriceTable | None = None,
) -> MetricSet:
    """Reduce a configuration's outcomes to one set of metrics.

    Args:
        outcomes: Every answered question for one configuration.
        model: Model name, used to price the token counts.
        prices: Price table; everything is free when omitted.

    Returns:
        The aggregated metrics. An empty input yields a zeroed set rather than an error,
        so a configuration that failed outright still reports something.
    """
    if not outcomes:
        return MetricSet(question_count=0)

    table = prices or PriceTable()
    retrieval = [o.answer.retrieval_ms for o in outcomes]
    generation = [o.answer.generation_ms for o in outcomes]
    total_tokens = sum(o.answer.usage.total_tokens for o in outcomes)
    cost = sum(
        table.cost(model, o.answer.usage.prompt_tokens, o.answer.usage.completion_tokens)
        for o in outcomes
    )

    return MetricSet(
        question_count=len(outcomes),
        hit_rate=hit_rate(outcomes),
        mrr=mean_reciprocal_rank(outcomes),
        abstention_accuracy=abstention_accuracy(outcomes),
        faithfulness=_mean_score(outcomes, "faithfulness"),
        answer_relevancy=_mean_score(outcomes, "answer_relevancy"),
        context_precision=_mean_score(outcomes, "context_precision"),
        context_recall=_mean_score(outcomes, "context_recall"),
        p50_retrieval_ms=median(retrieval) if retrieval else None,
        p95_retrieval_ms=percentile(retrieval, _P95),
        p50_generation_ms=median(generation) if generation else None,
        p95_generation_ms=percentile(generation, _P95),
        total_tokens=total_tokens,
        estimated_cost_usd=cost,
    )


def aggregate_by_difficulty(
    outcomes: Sequence[QuestionOutcome],
    *,
    model: str = "",
    prices: PriceTable | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Recompute the metrics separately within each difficulty band.

    Args:
        outcomes: Every answered question for one configuration.
        model: Model name, used to price the token counts.
        prices: Price table; everything is free when omitted.

    Returns:
        Difficulty name mapped to that band's metrics.
    """
    bands: dict[Difficulty, list[QuestionOutcome]] = {}
    for outcome in outcomes:
        bands.setdefault(outcome.question.difficulty, []).append(outcome)
    return {
        band.value: aggregate(items, model=model, prices=prices).as_dict()
        for band, items in bands.items()
    }


def _mean_score(outcomes: Sequence[QuestionOutcome], name: str) -> float | None:
    """Mean of one per-question score, ignoring questions it was not computed for."""
    values = [o.scores[name] for o in outcomes if name in o.scores]
    return sum(values) / len(values) if values else None
