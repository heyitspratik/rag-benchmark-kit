"""The migration is the schema of record, so it is verified, not assumed."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

EXPECTED_TABLES = {
    "benchmark_runs",
    "configuration_metrics",
    "question_results",
    "run_configurations",
}


def _config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.cmd_opts = None
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["x_arguments"] = [f"url={database_url}"]
    return config


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'migrations.db'}"


def test_there_is_exactly_one_head(monkeypatch: pytest.MonkeyPatch) -> None:
    # More than one head means two migrations branched and someone must merge them
    # before `alembic upgrade head` is unambiguous.
    monkeypatch.setenv("POSTGRES_DSN", "sqlite://")
    scripts = ScriptDirectory.from_config(Config("alembic.ini"))

    assert len(scripts.get_heads()) == 1


def test_upgrade_creates_every_table(database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_DSN", database_url)

    command.upgrade(_config(database_url), "head")

    engine = create_engine(database_url)
    try:
        assert set(inspect(engine).get_table_names()) >= EXPECTED_TABLES
    finally:
        engine.dispose()


def test_component_columns_are_indexed(database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Grouping results by component is the query this schema exists to serve.
    monkeypatch.setenv("POSTGRES_DSN", database_url)
    command.upgrade(_config(database_url), "head")

    engine = create_engine(database_url)
    try:
        indexed = {
            column
            for index in inspect(engine).get_indexes("run_configurations")
            for column in index["column_names"]
        }
    finally:
        engine.dispose()

    assert {"chunker", "embedder", "retriever", "store", "generator"} <= indexed


def test_downgrade_reverts_cleanly(database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_DSN", database_url)
    config = _config(database_url)
    command.upgrade(config, "head")

    command.downgrade(config, "base")

    engine = create_engine(database_url)
    try:
        remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()

    assert remaining == set()


def test_the_models_and_the_migration_agree(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If these drift, a migration that passes locally fails against a real database.
    monkeypatch.setenv("POSTGRES_DSN", database_url)
    config = _config(database_url)
    command.upgrade(config, "head")

    command.check(config)


def test_upgrading_a_populated_database_succeeds(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Adding a NOT NULL column to a table that already holds rows fails without a server
    # default, and that failure only appears against a database with real data in it.
    import uuid

    from sqlalchemy import text

    monkeypatch.setenv("POSTGRES_DSN", database_url)
    config = _config(database_url)
    command.upgrade(config, "0001")

    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO benchmark_runs (id, name, git_sha, git_dirty, status, "
                    "eval_set, question_count, sweep_config) VALUES "
                    "(:i, 'old', 'a', '0', 'completed', 'e.jsonl', 1, '{}')"
                ),
                {"i": str(uuid.uuid4())},
            )
    finally:
        engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("benchmark_runs")}
    finally:
        engine.dispose()

    assert {"llm_provider", "llm_model"} <= columns
