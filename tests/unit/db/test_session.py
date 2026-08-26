import pytest
from sqlalchemy import Engine, func, select, text
from sqlalchemy.orm import sessionmaker

from rag_bench.core.settings import DatabaseSettings
from rag_bench.db.models import BenchmarkRun
from rag_bench.db.session import (
    DatabaseError,
    build_engine,
    check_database,
    get_sessionmaker,
    session_scope,
)

from .conftest import make_run


def test_the_engine_uses_the_configured_dsn() -> None:
    settings = DatabaseSettings(_env_file=None, dsn="sqlite://")

    engine = build_engine(settings)

    assert engine.url.drivername == "sqlite"
    engine.dispose()


def test_a_successful_block_commits(engine: Engine) -> None:
    with session_scope(engine) as session:
        session.add(make_run())

    with sessionmaker(bind=engine)() as check:
        assert check.scalar(select(func.count()).select_from(BenchmarkRun)) == 1


def test_a_failing_block_rolls_back(engine: Engine) -> None:
    # A configuration that dies mid-run must not leave a half-written row for the next
    # one to trip over.
    with pytest.raises(RuntimeError), session_scope(engine) as session:
        session.add(make_run())
        session.flush()
        raise RuntimeError("boom")

    with sessionmaker(bind=engine)() as check:
        assert check.scalar(select(func.count()).select_from(BenchmarkRun)) == 0


def test_a_database_failure_becomes_a_project_error(engine: Engine) -> None:
    with pytest.raises(DatabaseError, match="transaction failed"), session_scope(engine) as session:
        session.execute(text("SELECT * FROM a_table_that_does_not_exist"))


def test_the_session_is_closed_either_way(engine: Engine) -> None:
    with session_scope(engine) as session:
        session.add(make_run())

    assert not session.is_active or not session.in_transaction()


def test_the_session_factory_binds_to_the_given_engine(engine: Engine) -> None:
    factory = get_sessionmaker(engine)

    with factory() as session:
        assert session.get_bind() is engine


def test_check_database_passes_against_a_live_database(engine: Engine) -> None:
    check_database(engine)


def test_check_database_names_the_commands_that_fix_it() -> None:
    unreachable = build_engine(
        DatabaseSettings(_env_file=None, dsn="postgresql+psycopg://x@127.0.0.1:1/none")
    )
    try:
        with pytest.raises(DatabaseError, match="make migrate"):
            check_database(unreachable)
    finally:
        unreachable.dispose()
