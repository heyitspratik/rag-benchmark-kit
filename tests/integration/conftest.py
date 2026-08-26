"""Fixtures for tests that need a real Postgres with pgvector.

These are marked ``integration`` and excluded from the default run. They skip rather
than fail when no database is configured, so a contributor without Docker still gets a
green suite.
"""

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

DEFAULT_DSN = "postgresql+psycopg://rag:rag@localhost:5432/rag_bench"


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    engine = create_engine(os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        engine.dispose()
        pytest.skip(f"No Postgres available: {exc}")
    yield engine
    engine.dispose()
