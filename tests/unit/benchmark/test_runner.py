from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import Engine, func, select

from rag_bench.benchmark.runner import BenchmarkRunner, resume
from rag_bench.core.config import SweepConfig
from rag_bench.core.exceptions import ResourceNotFoundError
from rag_bench.db.models import (
    BenchmarkRun,
    ConfigurationMetrics,
    QuestionResult,
    RunConfiguration,
    RunStatus,
)
from rag_bench.db.session import session_scope

from .conftest import StubChatModel


@pytest.fixture
def offline(chat_model: StubChatModel) -> Iterator[None]:
    """Stub the provider at the factory the generator really calls."""
    with (
        patch("rag_bench.pipeline.querier.check_llm_health"),
        patch("rag_bench.core.llm.build_chat_model", return_value=chat_model),
    ):
        yield


@pytest.fixture
def runner(sweep: SweepConfig, bench_engine: Engine, offline: None) -> BenchmarkRunner:
    return BenchmarkRunner(sweep, engine=bench_engine, pricing=None, check_provider=False)


def test_a_run_records_every_configuration(runner: BenchmarkRunner, bench_engine: Engine) -> None:
    summary = runner.run()

    assert summary.completed == 4
    assert summary.failed == 0
    with session_scope(bench_engine) as session:
        assert session.scalar(select(func.count()).select_from(RunConfiguration)) == 4


def test_each_index_is_built_once_for_the_group_that_shares_it(
    runner: BenchmarkRunner,
) -> None:
    # Four configurations, two chunkers, so two ingestions rather than four.
    summary = runner.run()

    assert summary.indexes_built == 2


def test_the_run_records_the_commit_it_was_produced_by(
    runner: BenchmarkRunner, bench_engine: Engine
) -> None:
    run_id = runner.run().run_id

    with session_scope(bench_engine) as session:
        run = session.get(BenchmarkRun, run_id)
        assert run is not None
        assert len(run.git_sha) == 40
        assert run.status == RunStatus.COMPLETED


def test_every_question_is_answered_for_every_configuration(
    runner: BenchmarkRunner, bench_engine: Engine
) -> None:
    runner.run()

    with session_scope(bench_engine) as session:
        assert session.scalar(select(func.count()).select_from(QuestionResult)) == 12


def test_metrics_are_stored_per_configuration(
    runner: BenchmarkRunner, bench_engine: Engine
) -> None:
    runner.run()

    with session_scope(bench_engine) as session:
        rows = session.scalars(select(ConfigurationMetrics)).all()

    assert len(rows) == 4
    assert all(row.question_count == 3 for row in rows)
    assert all(row.hit_rate is not None for row in rows)


def test_the_difficulty_breakdown_is_persisted(
    runner: BenchmarkRunner, bench_engine: Engine
) -> None:
    runner.run()

    with session_scope(bench_engine) as session:
        metrics = session.scalars(select(ConfigurationMetrics)).first()

    assert metrics is not None
    assert {"single_hop", "multi_hop", "negative"} <= set(metrics.by_difficulty)


def test_an_abstention_is_recorded(runner: BenchmarkRunner, bench_engine: Engine) -> None:
    runner.run()

    with session_scope(bench_engine) as session:
        declined = session.scalars(
            select(QuestionResult).where(QuestionResult.question_id == "q3")
        ).all()

    assert declined
    assert all(row.abstained for row in declined)


def test_results_are_written_as_the_run_progresses(
    runner: BenchmarkRunner, bench_engine: Engine
) -> None:
    # A grid takes hours; a crash at configuration 19 must not discard the first 18.
    run_id = runner.start()
    with session_scope(bench_engine) as session:
        assert session.get(BenchmarkRun, run_id) is not None
        assert session.scalar(select(func.count()).select_from(QuestionResult)) == 0

    runner.execute(run_id)

    with session_scope(bench_engine) as session:
        assert session.scalar(select(func.count()).select_from(QuestionResult)) == 12


@pytest.mark.usefixtures("offline")
def test_resuming_a_finished_run_does_nothing(
    runner: BenchmarkRunner, bench_engine: Engine
) -> None:
    run_id = runner.run().run_id

    again = resume(run_id, engine=bench_engine, pricing=None, check_provider=False)

    assert again.completed == 0
    assert again.skipped == 4
    assert again.indexes_built == 0


@pytest.mark.usefixtures("offline")
def test_resuming_redoes_only_the_interrupted_configuration(
    runner: BenchmarkRunner, bench_engine: Engine
) -> None:
    run_id = runner.run().run_id
    with session_scope(bench_engine) as session:
        victim = session.scalars(select(RunConfiguration)).first()
        assert victim is not None
        victim.status = RunStatus.FAILED

    again = resume(run_id, engine=bench_engine, pricing=None, check_provider=False)

    assert again.completed == 1
    assert again.skipped == 3
    # Only the index that configuration needs is rebuilt, not both.
    assert again.indexes_built == 1


@pytest.mark.usefixtures("offline")
def test_resuming_does_not_duplicate_question_results(
    runner: BenchmarkRunner, bench_engine: Engine
) -> None:
    run_id = runner.run().run_id
    with session_scope(bench_engine) as session:
        victim = session.scalars(select(RunConfiguration)).first()
        assert victim is not None
        victim.status = RunStatus.FAILED

    resume(run_id, engine=bench_engine, pricing=None, check_provider=False)

    with session_scope(bench_engine) as session:
        assert session.scalar(select(func.count()).select_from(QuestionResult)) == 12


def test_resuming_an_unknown_run_is_reported(bench_engine: Engine) -> None:
    from uuid import uuid4

    with pytest.raises(ResourceNotFoundError, match="No benchmark run"):
        resume(uuid4(), engine=bench_engine, pricing=None, check_provider=False)


@pytest.mark.usefixtures("offline")
def test_a_broken_configuration_does_not_abort_the_sweep(
    sweep: SweepConfig, bench_engine: Engine
) -> None:
    # One bad combination should not discard the results of the others.
    broken = sweep.model_copy(
        update={"sweep": {"chunker": ["fixed", "structural"], "retriever": ["dense", "nope"]}}
    )
    runner = BenchmarkRunner(broken, engine=bench_engine, pricing=None, check_provider=False)

    summary = runner.run()

    assert summary.completed == 2
    assert summary.failed == 2
    with session_scope(bench_engine) as session:
        failed = session.scalars(
            select(RunConfiguration).where(RunConfiguration.status == RunStatus.FAILED)
        ).all()
    assert all("UNKNOWN_COMPONENT" in (row.error or "") for row in failed)


@pytest.mark.usefixtures("offline")
def test_the_eval_set_can_be_overridden(
    sweep: SweepConfig, bench_engine: Engine, workspace: Path
) -> None:
    smaller = workspace / "one.jsonl"
    smaller.write_text(
        '{"id": "only", "question": "What fee?", "ground_truth": "A fee.", '
        '"source_refs": ["controllers.md#Fees"], "difficulty": "single_hop"}\n'
    )

    runner = BenchmarkRunner(
        sweep, engine=bench_engine, pricing=None, eval_set=smaller, check_provider=False
    )

    assert len(runner.eval_set) == 1


def test_ragas_scores_reach_both_the_rows_and_the_aggregate(
    sweep: SweepConfig, bench_engine: Engine, offline: None
) -> None:
    from dataclasses import replace as replace_outcome
    from unittest.mock import MagicMock

    scorer = MagicMock()
    scorer.score.side_effect = lambda outcomes: [
        replace_outcome(o, scores={"faithfulness": 0.75}) for o in outcomes
    ]
    runner = BenchmarkRunner(
        sweep, engine=bench_engine, pricing=None, check_provider=False, scorer=scorer
    )

    runner.run()

    with session_scope(bench_engine) as session:
        metrics = session.scalars(select(ConfigurationMetrics)).all()
        results = session.scalars(select(QuestionResult)).all()

    assert all(row.faithfulness == pytest.approx(0.75) for row in metrics)
    assert all(row.scores.get("faithfulness") == 0.75 for row in results)


def test_a_scoring_failure_keeps_the_deterministic_metrics(
    sweep: SweepConfig, bench_engine: Engine, offline: None
) -> None:
    # The answers were expensive to produce; a failed judge must not discard them.
    from unittest.mock import MagicMock

    from rag_bench.core.exceptions import BenchmarkError

    scorer = MagicMock()
    scorer.score.side_effect = BenchmarkError("judge unavailable")
    runner = BenchmarkRunner(
        sweep, engine=bench_engine, pricing=None, check_provider=False, scorer=scorer
    )

    summary = runner.run()

    assert summary.completed == 4
    with session_scope(bench_engine) as session:
        metrics = session.scalars(select(ConfigurationMetrics)).all()
    assert all(row.faithfulness is None for row in metrics)
    assert all(row.hit_rate is not None for row in metrics)


def test_the_run_records_which_model_generated_the_answers(
    runner: BenchmarkRunner, bench_engine: Engine
) -> None:
    # A results table that does not say which model produced it cannot be reproduced,
    # for the same reason the commit is recorded.
    run_id = runner.run().run_id

    with session_scope(bench_engine) as session:
        run = session.get(BenchmarkRun, run_id)
        assert run is not None
        assert run.llm_provider == "ollama"
        assert run.llm_model == "llama3.2:3b"
