from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from rag_bench.db.models import (
    BenchmarkRun,
    ConfigurationMetrics,
    RunConfiguration,
    RunStatus,
)
from rag_bench.db.session import session_scope


def _seed(engine: Engine, count: int) -> list[UUID]:
    """Insert runs with distinct creation times, newest last."""
    ids: list[UUID] = []
    base = datetime(2026, 1, 1, tzinfo=UTC)
    with session_scope(engine) as session:
        for index in range(count):
            run = BenchmarkRun(
                name=f"run_{index:02d}",
                git_sha=f"{index:040d}",
                status=RunStatus.COMPLETED,
                eval_set="data/eval/smoke.jsonl",
                question_count=10,
                llm_provider="ollama",
                llm_model="llama3.2:3b",
                sweep_config={"chunker": ["structural"]},
                created_at=base + timedelta(minutes=index),
            )
            session.add(run)
            session.flush()
            ids.append(run.id)
    return ids


@pytest.fixture
def seeded(api_engine: Engine) -> list[UUID]:
    return _seed(api_engine, 5)


def test_runs_are_listed_newest_first(client: TestClient, seeded: list[UUID]) -> None:
    body = client.get("/api/v1/benchmark-runs").json()

    assert [item["name"] for item in body["items"]] == [
        "run_04",
        "run_03",
        "run_02",
        "run_01",
        "run_00",
    ]


def test_the_list_uses_the_shared_envelope(client: TestClient, seeded: list[UUID]) -> None:
    body = client.get("/api/v1/benchmark-runs").json()

    assert set(body) == {"items", "next_cursor"}


def test_a_full_page_offers_a_cursor_and_the_last_does_not(
    client: TestClient, seeded: list[UUID]
) -> None:
    first = client.get("/api/v1/benchmark-runs", params={"limit": 2}).json()

    assert first["next_cursor"] is not None

    last = client.get("/api/v1/benchmark-runs", params={"limit": 50}).json()
    assert last["next_cursor"] is None


def test_paging_visits_every_run_exactly_once(client: TestClient, seeded: list[UUID]) -> None:
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(10):
        params = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        body = client.get("/api/v1/benchmark-runs", params=params).json()
        seen.extend(item["name"] for item in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break

    assert seen == ["run_04", "run_03", "run_02", "run_01", "run_00"]
    assert len(seen) == len(set(seen))


def test_a_run_inserted_mid_paging_does_not_shift_the_page(
    client: TestClient, api_engine: Engine, seeded: list[UUID]
) -> None:
    # This is why the cursor is keyset rather than an offset: an insert between pages
    # would otherwise make an offset skip or repeat a row.
    first = client.get("/api/v1/benchmark-runs", params={"limit": 2}).json()
    _seed(api_engine, 1)

    second = client.get(
        "/api/v1/benchmark-runs", params={"limit": 2, "cursor": first["next_cursor"]}
    ).json()

    assert [item["name"] for item in second["items"]] == ["run_02", "run_01"]


def test_a_malformed_cursor_is_reported(client: TestClient, seeded: list[UUID]) -> None:
    response = client.get("/api/v1/benchmark-runs", params={"cursor": "not-a-cursor"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CONFIG_INVALID"


def test_an_out_of_range_limit_is_rejected(client: TestClient) -> None:
    assert client.get("/api/v1/benchmark-runs", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/benchmark-runs", params={"limit": 1000}).status_code == 422


def test_a_run_detail_includes_its_configurations_and_metrics(
    client: TestClient, api_engine: Engine
) -> None:
    run_id = _seed(api_engine, 1)[0]
    with session_scope(api_engine) as session:
        configuration = RunConfiguration(
            run_id=run_id,
            fingerprint="f" * 16,
            index_fingerprint="i" * 16,
            corpus="eu_regulations",
            chunker="structural",
            embedder="bge_small",
            store="qdrant",
            retriever="hybrid",
            generator="cited",
            resolved_config={},
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

    body = client.get(f"/api/v1/benchmark-runs/{run_id}").json()

    assert body["llm_model"] == "llama3.2:3b"
    assert len(body["configurations"]) == 1
    metrics = body["configurations"][0]["metrics"]
    assert metrics["hit_rate"] == 0.9
    assert metrics["by_difficulty"]["multi_hop"]["hit_rate"] == 0.6


def test_a_configuration_without_metrics_reports_null(
    client: TestClient, api_engine: Engine
) -> None:
    run_id = _seed(api_engine, 1)[0]
    with session_scope(api_engine) as session:
        session.add(
            RunConfiguration(
                run_id=run_id,
                fingerprint="f" * 16,
                index_fingerprint="i" * 16,
                corpus="c",
                chunker="fixed",
                embedder="bge_small",
                store="qdrant",
                retriever="dense",
                generator="cited",
                resolved_config={},
                status=RunStatus.FAILED,
                error="UNKNOWN_COMPONENT: nope",
            )
        )

    body = client.get(f"/api/v1/benchmark-runs/{run_id}").json()

    assert body["configurations"][0]["metrics"] is None
    assert body["configurations"][0]["error"].startswith("UNKNOWN_COMPONENT")


def test_an_unknown_run_is_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/benchmark-runs/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_an_empty_database_lists_nothing_rather_than_failing(client: TestClient) -> None:
    body = client.get("/api/v1/benchmark-runs").json()

    assert body == {"items": [], "next_cursor": None}
