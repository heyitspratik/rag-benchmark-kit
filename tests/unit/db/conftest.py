"""A throwaway SQLite database per test.

``create_all`` appears here and nowhere else: production schema is owned by Alembic, and
a schema built two different ways drifts. A separate test verifies the migration and the
models still agree, which is what makes using create_all here safe.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from rag_bench.db.models import Base, BenchmarkRun, RunConfiguration, RunStatus


@pytest.fixture
def engine() -> Iterator[Engine]:
    instance = create_engine("sqlite://")

    # SQLite ignores foreign keys unless asked, and the cascade behaviour these models
    # rely on would go untested otherwise.
    @event.listens_for(instance, "connect")
    def _enable_foreign_keys(connection, _record) -> None:  # type: ignore[no-untyped-def]
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(instance)
    yield instance
    instance.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as open_session:
        yield open_session


def make_run(**overrides: object) -> BenchmarkRun:
    """Build a benchmark run with sensible defaults."""
    fields: dict[str, object] = {
        "name": "full_grid_v1",
        "git_sha": "a" * 40,
        "status": RunStatus.RUNNING,
        "eval_set": "data/eval/smoke.jsonl",
        "question_count": 10,
        "sweep_config": {"chunker": ["fixed", "structural"]},
        "started_at": datetime.now(UTC),
    }
    return BenchmarkRun(**(fields | overrides))


def make_configuration(run: BenchmarkRun, **overrides: object) -> RunConfiguration:
    """Build a configuration attached to a run."""
    fields: dict[str, object] = {
        "run_id": run.id or uuid.uuid4(),
        "fingerprint": "f" * 16,
        "index_fingerprint": "i" * 16,
        "corpus": "eu_regulations",
        "chunker": "structural",
        "embedder": "bge_small",
        "store": "qdrant",
        "retriever": "hybrid_rerank",
        "generator": "cited",
        "resolved_config": {"chunker": {"name": "structural"}},
        "status": RunStatus.PENDING,
    }
    return RunConfiguration(**(fields | overrides))
