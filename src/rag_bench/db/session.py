"""Engine and session management.

The schema is owned by Alembic. Nothing here calls ``create_all``: a schema created two
different ways drifts, and the moment it does, a migration that works locally fails in
production.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from rag_bench.core.exceptions import RagBenchError
from rag_bench.core.logging import get_logger
from rag_bench.core.settings import DatabaseSettings, get_settings

logger = get_logger(__name__)


class DatabaseError(RagBenchError):
    """The database is unreachable or rejected a statement."""

    code = "DATABASE_ERROR"
    http_status = 503


def build_engine(settings: DatabaseSettings | None = None) -> Engine:
    """Create an engine for the configured database.

    Args:
        settings: Database settings; read from the environment when omitted.

    Returns:
        A new engine. Callers that want the shared one should use :func:`get_engine`.
    """
    resolved = settings or get_settings().db
    return create_engine(
        resolved.dsn,
        echo=resolved.echo,
        pool_size=resolved.pool_size,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use."""
    return build_engine()


def get_sessionmaker(engine: Engine | None = None) -> sessionmaker[Session]:
    """Return a session factory bound to an engine.

    Args:
        engine: Engine to bind to; the shared one when omitted.

    Returns:
        A configured session factory.
    """
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Run a unit of work in a transaction, committing on success.

    Args:
        engine: Engine to bind to; the shared one when omitted.

    Yields:
        An open session.

    Raises:
        DatabaseError: If the transaction failed. The session is always rolled back and
            closed first, so a failed benchmark configuration cannot leave a half-written
            row behind for the next one to trip over.
    """
    session = get_sessionmaker(engine)()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        raise DatabaseError(f"Database transaction failed: {exc}") from exc
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database(engine: Engine | None = None) -> None:
    """Verify the database answers, failing fast with an actionable message.

    Args:
        engine: Engine to test; the shared one when omitted.

    Raises:
        DatabaseError: If the database cannot be reached.
    """
    try:
        with (engine or get_engine()).connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise DatabaseError(
            f"Cannot reach the database: {exc}. Start it with `make up`, then run `make migrate`."
        ) from exc
