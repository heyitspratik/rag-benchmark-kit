import pytest

from rag_bench.benchmark.evalset import Difficulty, EvalQuestion
from rag_bench.benchmark.metrics import (
    PriceTable,
    QuestionOutcome,
    abstention_accuracy,
    aggregate,
    aggregate_by_difficulty,
    hit_rate,
    mean_reciprocal_rank,
    percentile,
)
from rag_bench.core.models import Answer, Chunk, ScoredChunk, TokenUsage


def _question(qid: str, refs: tuple[str, ...], difficulty: str = "single_hop") -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        question="Q?",
        ground_truth="A",
        source_refs=refs,
        difficulty=Difficulty(difficulty),
    )


def _answer(
    ref_ranks: list[tuple[str, ...]],
    *,
    abstained: bool = False,
    retrieval_ms: float = 10.0,
    generation_ms: float = 100.0,
    usage: TokenUsage | None = None,
) -> Answer:
    contexts = tuple(
        ScoredChunk(
            chunk=Chunk.create(
                doc_id="d", ordinal=rank, text=f"chunk {rank}", char_start=0, char_end=5
            ).with_sections(refs),
            score=1.0 - rank / 10,
            rank=rank,
        )
        for rank, refs in enumerate(ref_ranks)
    )
    return Answer(
        question="Q?",
        text="A",
        abstained=abstained,
        contexts=contexts,
        usage=usage or TokenUsage(),
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )


def _outcome(
    refs: tuple[str, ...], ranks: list[tuple[str, ...]], **kwargs: object
) -> QuestionOutcome:
    difficulty = str(kwargs.pop("difficulty", "single_hop"))
    return QuestionOutcome(
        question=_question("q", refs, difficulty),
        answer=_answer(ranks, **kwargs),  # type: ignore[arg-type]
    )


def test_a_hit_needs_only_one_retrieved_chunk_to_match() -> None:
    outcome = _outcome(("GDPR Art. 5",), [("GDPR Art. 9",), ("GDPR Art. 5",)])

    assert outcome.is_hit is True
    assert hit_rate([outcome]) == 1.0


def test_a_miss_is_recorded_when_nothing_matches() -> None:
    outcome = _outcome(("GDPR Art. 5",), [("GDPR Art. 9",)])

    assert outcome.is_hit is False
    assert hit_rate([outcome]) == 0.0


def test_negative_questions_are_excluded_from_hit_rate() -> None:
    # A negative question has no correct section, so counting it would penalise the
    # system for behaving correctly.
    hit = _outcome(("GDPR Art. 5",), [("GDPR Art. 5",)])
    negative = _outcome((), [("GDPR Art. 9",)], difficulty="negative")

    assert hit_rate([hit, negative]) == 1.0


def test_hit_rate_is_none_when_nothing_is_answerable() -> None:
    assert hit_rate([_outcome((), [], difficulty="negative")]) is None
    assert hit_rate([]) is None


def test_reciprocal_rank_rewards_an_earlier_hit() -> None:
    first = _outcome(("A",), [("A",), ("B",)])
    third = _outcome(("A",), [("B",), ("C",), ("A",)])

    assert mean_reciprocal_rank([first]) == pytest.approx(1.0)
    assert mean_reciprocal_rank([third]) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank([first, third]) == pytest.approx((1.0 + 1 / 3) / 2)


def test_a_miss_contributes_zero_to_reciprocal_rank() -> None:
    hit = _outcome(("A",), [("A",)])
    miss = _outcome(("A",), [("B",)])

    assert mean_reciprocal_rank([hit, miss]) == pytest.approx(0.5)


def test_abstention_accuracy_counts_only_negatives() -> None:
    declined = _outcome((), [], abstained=True, difficulty="negative")
    answered = _outcome((), [], abstained=False, difficulty="negative")
    unrelated = _outcome(("A",), [("A",)], abstained=False)

    assert abstention_accuracy([declined, answered, unrelated]) == pytest.approx(0.5)


def test_abstention_accuracy_is_none_without_negatives() -> None:
    assert abstention_accuracy([_outcome(("A",), [("A",)])]) is None


def test_percentiles_use_nearest_rank() -> None:
    values = [float(n) for n in range(1, 101)]

    assert percentile(values, 0.5) == 51.0
    assert percentile(values, 0.95) == 96.0
    assert percentile([], 0.95) is None


def test_percentile_of_a_single_sample_is_that_sample() -> None:
    assert percentile([7.0], 0.95) == 7.0


def test_aggregate_reports_latency_and_tokens() -> None:
    outcomes = [
        _outcome(
            ("A",),
            [("A",)],
            retrieval_ms=10.0,
            generation_ms=100.0,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=10),
        ),
        _outcome(
            ("A",),
            [("A",)],
            retrieval_ms=30.0,
            generation_ms=300.0,
            usage=TokenUsage(prompt_tokens=200, completion_tokens=20),
        ),
    ]

    metrics = aggregate(outcomes)

    assert metrics.question_count == 2
    assert metrics.p50_retrieval_ms == pytest.approx(20.0)
    assert metrics.p95_generation_ms == pytest.approx(300.0)
    assert metrics.total_tokens == 330


def test_an_empty_configuration_aggregates_to_zero_rather_than_failing() -> None:
    # A configuration that failed outright must still report something.
    metrics = aggregate([])

    assert metrics.question_count == 0
    assert metrics.hit_rate is None


def test_llm_judged_metrics_come_from_per_question_scores() -> None:
    scored = [
        QuestionOutcome(_question("a", ("A",)), _answer([("A",)]), {"faithfulness": 0.8}),
        QuestionOutcome(_question("b", ("A",)), _answer([("A",)]), {"faithfulness": 0.6}),
    ]

    assert aggregate(scored).faithfulness == pytest.approx(0.7)


def test_a_metric_nobody_scored_stays_none() -> None:
    assert aggregate([_outcome(("A",), [("A",)])]).faithfulness is None


def test_a_local_model_costs_nothing() -> None:
    # The default provider runs on the machine, so zero is the truthful figure.
    table = PriceTable({"gpt-4o-mini": {"input_usd_per_1k": 0.00015}})

    assert table.cost("llama3.2:3b", 10_000, 1_000) == 0.0


def test_cost_is_priced_per_thousand_tokens() -> None:
    table = PriceTable({"gpt-4o-mini": {"input_usd_per_1k": 0.001, "output_usd_per_1k": 0.002}})

    assert table.cost("gpt-4o-mini", 2000, 1000) == pytest.approx(0.004)


def test_metrics_are_recomputed_within_each_difficulty_band() -> None:
    # A configuration that wins overall but collapses on multi-hop is the finding worth
    # reading, and the aggregate hides it.
    outcomes = [
        _outcome(("A",), [("A",)], difficulty="single_hop"),
        _outcome(("A",), [("B",)], difficulty="multi_hop"),
        _outcome((), [], abstained=True, difficulty="negative"),
    ]

    bands = aggregate_by_difficulty(outcomes)

    assert bands["single_hop"]["hit_rate"] == 1.0
    assert bands["multi_hop"]["hit_rate"] == 0.0
    assert bands["negative"]["abstention_accuracy"] == 1.0


def test_the_metric_dict_covers_every_reported_field() -> None:
    keys = set(aggregate([]).as_dict())

    assert {"hit_rate", "mrr", "abstention_accuracy", "faithfulness", "estimated_cost_usd"} <= keys
