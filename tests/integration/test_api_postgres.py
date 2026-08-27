"""The API against a real Postgres, with the schema applied by Alembic.

The unit tests run on SQLite, which forgives things Postgres does not: JSONB, UUID
columns and timezone-aware timestamps all behave differently. These check the API reads
correctly from the database it will actually run against.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from rag_bench.api.main import create_app
from rag_bench.core.settings import Settings
from rag_bench.db.models import (
    BenchmarkRun,
    ConfigurationMetrics,
    RunConfiguration,
    RunStatus,
)
from rag_bench.db.session import session_scope

pytestmark = pytest.mark.integration


@pytest.fixture
def migrated(postgres_engine: Engine) -> Iterator[Engine]:
    """Apply the migrations, then leave the schema clean for the next test."""
    url = str(postgres_engine.url.render_as_string(hide_password=False))
    config = Config("alembic.ini")
    config.cmd_opts = None
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["x_arguments"] = [f"url={url}"]

    with postgres_engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    command.upgrade(config, "head")
    yield postgres_engine
    with postgres_engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))


@pytest.fixture
def client(migrated: Engine) -> Iterator[TestClient]:
    app = create_app(settings=Settings(_env_file=None), engine=migrated)
    with TestClient(app) as open_client:
        yield open_client


def _seed(engine: Engine) -> UUID:
    """Insert one run with a configuration and its metrics."""
    with session_scope(engine) as session:
        run = BenchmarkRun(
            name="integration",
            git_sha="a" * 40,
            status=RunStatus.COMPLETED,
            eval_set="data/eval/smoke.jsonl",
            question_count=10,
            llm_provider="ollama",
            llm_model="llama3.2:3b",
            sweep_config={"chunker": ["structural", "fixed"]},
            started_at=datetime.now(UTC),
        )
        session.add(run)
        session.flush()
        configuration = RunConfiguration(
            run_id=run.id,
            fingerprint="f" * 16,
            index_fingerprint="i" * 16,
            corpus="eu_regulations",
            chunker="structural",
            embedder="bge_small",
            store="pgvector",
            retriever="hybrid",
            generator="cited",
            resolved_config={"chunker": {"name": "structural"}},
            status=RunStatus.COMPLETED,
        )
        session.add(configuration)
        session.flush()
        session.add(
            ConfigurationMetrics(
                configuration_id=configuration.id,
                hit_rate=0.9,
                mrr=0.75,
                question_count=10,
                by_difficulty={"multi_hop": {"hit_rate": 0.6}},
            )
        )
        return run.id


def test_readiness_reports_a_reachable_database(client: TestClient) -> None:
    checks = client.get("/health/ready").json()["checks"]

    assert checks["database"] == "ok"


def test_a_run_round_trips_through_postgres(client: TestClient, migrated: Engine) -> None:
    run_id = _seed(migrated)

    body = client.get(f"/api/v1/benchmark-runs/{run_id}").json()

    assert body["name"] == "integration"
    assert body["llm_model"] == "llama3.2:3b"
    assert body["configurations"][0]["metrics"]["hit_rate"] == 0.9


def test_jsonb_columns_survive_the_round_trip(client: TestClient, migrated: Engine) -> None:
    # JSONB is the column type in production and plain JSON under SQLite, so the nested
    # difficulty breakdown is worth checking against the real one.
    run_id = _seed(migrated)

    body = client.get(f"/api/v1/benchmark-runs/{run_id}").json()

    assert body["configurations"][0]["metrics"]["by_difficulty"] == {"multi_hop": {"hit_rate": 0.6}}


def test_paging_works_against_postgres(client: TestClient, migrated: Engine) -> None:
    for _ in range(3):
        _seed(migrated)

    first = client.get("/api/v1/benchmark-runs", params={"limit": 2}).json()
    second = client.get(
        "/api/v1/benchmark-runs", params={"limit": 2, "cursor": first["next_cursor"]}
    ).json()

    assert len(first["items"]) == 2
    assert len(second["items"]) == 1
    ids = {item["id"] for item in first["items"] + second["items"]}
    assert len(ids) == 3


def test_an_unknown_run_is_404(client: TestClient) -> None:
    assert client.get(f"/api/v1/benchmark-runs/{uuid4()}").status_code == 404
