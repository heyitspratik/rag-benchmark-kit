import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rag_bench.db.models import (
    BenchmarkRun,
    ConfigurationMetrics,
    QuestionResult,
    RunConfiguration,
    RunStatus,
)

from .conftest import make_configuration, make_run


def _persisted_configuration(session: Session) -> RunConfiguration:
    run = make_run()
    session.add(run)
    session.flush()
    configuration = make_configuration(run)
    session.add(configuration)
    session.flush()
    return configuration


def test_a_run_gets_a_uuid_without_a_round_trip(session: Session) -> None:
    run = make_run()
    session.add(run)
    session.flush()

    assert isinstance(run.id, uuid.UUID)


def test_a_run_records_the_commit_that_produced_it(session: Session) -> None:
    # Results that cannot be traced to the code that made them are not reproducible.
    run = make_run(git_sha="b" * 40, git_dirty=True)
    session.add(run)
    session.commit()

    stored = session.scalars(select(BenchmarkRun)).one()
    assert stored.git_sha == "b" * 40
    assert stored.git_dirty is True


def test_created_at_is_set_by_the_database(session: Session) -> None:
    run = make_run()
    session.add(run)
    session.commit()
    session.refresh(run)

    assert run.created_at is not None


def test_sweep_config_round_trips_as_structured_data(session: Session) -> None:
    run = make_run(sweep_config={"chunker": ["fixed"], "embedder": ["bge_small"]})
    session.add(run)
    session.commit()

    stored = session.scalars(select(BenchmarkRun)).one()
    assert stored.sweep_config["chunker"] == ["fixed"]


def test_component_names_are_queryable_columns(session: Session) -> None:
    # "Which chunker won" must be a plain GROUP BY, not a JSON traversal.
    run = make_run()
    session.add(run)
    session.flush()
    for position, chunker in enumerate(("fixed", "fixed", "structural")):
        session.add(make_configuration(run, chunker=chunker, fingerprint=f"fp-{position}"))
    session.commit()

    grouped = dict(
        session.execute(
            select(RunConfiguration.chunker, func.count()).group_by(RunConfiguration.chunker)
        ).all()
    )
    assert grouped == {"fixed": 2, "structural": 1}


def test_a_configuration_cannot_be_registered_twice_in_one_run(session: Session) -> None:
    # This is what lets a resumed run tell finished work from work still to do.
    run = make_run()
    session.add(run)
    session.flush()
    session.add(make_configuration(run, fingerprint="same"))
    session.add(make_configuration(run, fingerprint="same"))

    with pytest.raises(IntegrityError):
        session.flush()


def test_the_same_fingerprint_may_appear_in_different_runs(session: Session) -> None:
    first, second = make_run(), make_run(name="other")
    session.add_all([first, second])
    session.flush()
    session.add(make_configuration(first, fingerprint="shared"))
    session.add(make_configuration(second, fingerprint="shared"))
    session.commit()

    assert session.scalar(select(func.count()).select_from(RunConfiguration)) == 2


def test_a_question_is_answered_once_per_configuration(session: Session) -> None:
    configuration = _persisted_configuration(session)
    for _ in range(2):
        session.add(
            QuestionResult(
                configuration_id=configuration.id,
                question_id="q_0042",
                difficulty="multi_hop",
                question="Why?",
                answer="Because.",
                retrieved_chunk_ids=[],
                retrieved_section_refs=[],
                cited_chunk_ids=[],
            )
        )

    with pytest.raises(IntegrityError):
        session.flush()


def test_a_question_result_stores_what_the_benchmark_scores(session: Session) -> None:
    configuration = _persisted_configuration(session)
    session.add(
        QuestionResult(
            configuration_id=configuration.id,
            question_id="q_0042",
            difficulty="multi_hop",
            category="data_subject_rights",
            question="May a fee be charged?",
            answer="A reasonable fee may be charged [1].",
            abstained=False,
            retrieved_chunk_ids=["chunk-a", "chunk-b"],
            retrieved_section_refs=["GDPR Art. 12(5)"],
            cited_chunk_ids=["chunk-a"],
            retrieval_ms=41.5,
            generation_ms=980.25,
            prompt_tokens=812,
            completion_tokens=44,
            scores={"faithfulness": 0.91},
        )
    )
    session.commit()

    stored = session.scalars(select(QuestionResult)).one()
    assert stored.retrieved_section_refs == ["GDPR Art. 12(5)"]
    assert stored.scores["faithfulness"] == 0.91
    assert stored.prompt_tokens + stored.completion_tokens == 856


def test_metrics_allow_missing_llm_judged_scores(session: Session) -> None:
    # RAGAS is itself LLM-judged and can fail without invalidating hit rate or latency.
    configuration = _persisted_configuration(session)
    session.add(
        ConfigurationMetrics(
            configuration_id=configuration.id,
            hit_rate=0.82,
            mrr=0.61,
            question_count=100,
            by_difficulty={"multi_hop": {"hit_rate": 0.55}},
        )
    )
    session.commit()

    stored = session.scalars(select(ConfigurationMetrics)).one()
    assert stored.faithfulness is None
    assert stored.hit_rate == 0.82
    assert stored.by_difficulty["multi_hop"]["hit_rate"] == 0.55


def test_one_metrics_row_per_configuration(session: Session) -> None:
    configuration = _persisted_configuration(session)
    session.add(ConfigurationMetrics(configuration_id=configuration.id))
    session.add(ConfigurationMetrics(configuration_id=configuration.id))

    with pytest.raises(IntegrityError):
        session.flush()


def test_deleting_a_run_removes_everything_beneath_it(session: Session) -> None:
    configuration = _persisted_configuration(session)
    session.add(
        QuestionResult(
            configuration_id=configuration.id,
            question_id="q_1",
            difficulty="single_hop",
            question="q",
            answer="a",
            retrieved_chunk_ids=[],
            retrieved_section_refs=[],
            cited_chunk_ids=[],
        )
    )
    session.add(ConfigurationMetrics(configuration_id=configuration.id))
    session.commit()

    session.delete(session.scalars(select(BenchmarkRun)).one())
    session.commit()

    assert session.scalar(select(func.count()).select_from(RunConfiguration)) == 0
    assert session.scalar(select(func.count()).select_from(QuestionResult)) == 0
    assert session.scalar(select(func.count()).select_from(ConfigurationMetrics)) == 0


def test_a_negative_question_count_is_rejected(session: Session) -> None:
    session.add(make_run(question_count=-1))

    with pytest.raises(IntegrityError):
        session.flush()


def test_status_round_trips_as_an_enum(session: Session) -> None:
    run = make_run(status=RunStatus.COMPLETED, finished_at=datetime.now(UTC))
    session.add(run)
    session.commit()

    stored = session.scalars(select(BenchmarkRun)).one()
    assert stored.status == RunStatus.COMPLETED
    assert stored.status == "completed"


def test_relationships_navigate_in_both_directions(session: Session) -> None:
    configuration = _persisted_configuration(session)
    session.commit()

    run = session.scalars(select(BenchmarkRun)).one()
    assert [c.id for c in run.configurations] == [configuration.id]
    assert run.configurations[0].run.id == run.id
