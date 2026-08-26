import pytest

from rag_bench.components.retrievers.fusion import (
    DEFAULT_RRF_K,
    RankedList,
    ReciprocalRankFusion,
)
from rag_bench.core.exceptions import ConfigValidationError


def test_the_score_follows_the_rrf_formula() -> None:
    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse([RankedList("dense", 1.0, ("a", "b"))])

    assert results[0].score == pytest.approx(1.0 / 61)
    assert results[1].score == pytest.approx(1.0 / 62)


def test_a_single_ranking_keeps_its_order() -> None:
    results = ReciprocalRankFusion().fuse([RankedList("dense", 1.0, ("a", "b", "c"))])

    assert [r.id for r in results] == ["a", "b", "c"]


def test_an_item_found_by_both_rankers_outranks_one_found_by_either() -> None:
    # The whole point of fusing: agreement between two different methods is evidence.
    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(
        [
            RankedList("dense", 1.0, ("solo_dense", "agreed")),
            RankedList("bm25", 1.0, ("solo_bm25", "agreed")),
        ]
    )

    assert results[0].id == "agreed"
    assert results[0].score == pytest.approx(1.0 / 62 + 1.0 / 62)


def test_weights_shift_the_balance_between_rankers() -> None:
    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(
        [
            RankedList("dense", 0.9, ("dense_top",)),
            RankedList("bm25", 0.1, ("bm25_top",)),
        ]
    )

    assert [r.id for r in results] == ["dense_top", "bm25_top"]


def test_a_zero_weighted_ranker_contributes_nothing() -> None:
    results = ReciprocalRankFusion().fuse(
        [
            RankedList("dense", 1.0, ("a",)),
            RankedList("bm25", 0.0, ("b",)),
        ]
    )

    assert [r.id for r in results] == ["a"]


def test_a_smaller_constant_lets_the_top_rank_dominate() -> None:
    # With k=0 the first rank scores 1.0 and the second 0.5; with k=60 they are within
    # two percent of each other. This is what the constant is for.
    sharp = ReciprocalRankFusion(k=0).fuse([RankedList("d", 1.0, ("a", "b"))])
    flat = ReciprocalRankFusion(k=60).fuse([RankedList("d", 1.0, ("a", "b"))])

    assert sharp[0].score / sharp[1].score == pytest.approx(2.0)
    assert flat[0].score / flat[1].score == pytest.approx(62 / 61)


def test_best_rank_is_recorded_across_rankers() -> None:
    results = ReciprocalRankFusion().fuse(
        [
            RankedList("dense", 1.0, ("x", "shared")),
            RankedList("bm25", 1.0, ("shared", "y")),
        ]
    )

    assert next(r for r in results if r.id == "shared").best_rank == 0


def test_ties_break_deterministically() -> None:
    # Equal scores must not reorder between runs, or a benchmark rerun would shuffle
    # results that are genuinely identical.
    rankings = [RankedList("a", 1.0, ("zebra",)), RankedList("b", 1.0, ("apple",))]

    first = [r.id for r in ReciprocalRankFusion().fuse(rankings)]
    second = [r.id for r in ReciprocalRankFusion().fuse(rankings)]

    assert first == second == ["apple", "zebra"]


def test_fusing_nothing_returns_nothing() -> None:
    assert ReciprocalRankFusion().fuse([]) == []
    assert ReciprocalRankFusion().fuse([RankedList("d", 1.0, ())]) == []


def test_the_default_constant_is_the_published_one() -> None:
    assert DEFAULT_RRF_K == 60
    assert ReciprocalRankFusion().k == 60


def test_a_negative_constant_is_rejected() -> None:
    with pytest.raises(ConfigValidationError, match="rrf_k must not be negative"):
        ReciprocalRankFusion(k=-1)
