from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_bench.core.exceptions import LLMProviderError
from rag_bench.db.session import DatabaseError


def test_liveness_ignores_dependencies(client: TestClient) -> None:
    # A dependency outage must never cause an orchestrator to restart a healthy process.
    with patch("rag_bench.api.v1.routes.health.check_database", side_effect=DatabaseError("down")):
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_passes_when_every_dependency_answers(client: TestClient) -> None:
    with (
        patch("rag_bench.api.v1.routes.health.check_database"),
        patch("rag_bench.api.v1.routes.health.check_llm_health"),
    ):
        response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"database": "ok", "llm": "ok"}


def test_readiness_reports_503_when_a_dependency_is_down(client: TestClient) -> None:
    # A load balancer should stop sending traffic rather than let requests fail one by one.
    with (
        patch("rag_bench.api.v1.routes.health.check_database"),
        patch(
            "rag_bench.api.v1.routes.health.check_llm_health",
            side_effect=LLMProviderError("ollama unreachable"),
        ),
    ):
        response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["llm"].startswith("unavailable")


def test_readiness_names_every_failing_dependency(client: TestClient) -> None:
    with (
        patch(
            "rag_bench.api.v1.routes.health.check_database",
            side_effect=DatabaseError("db down"),
        ),
        patch(
            "rag_bench.api.v1.routes.health.check_llm_health",
            side_effect=LLMProviderError("llm down"),
        ),
    ):
        checks = client.get("/health/ready").json()["checks"]

    assert all(value.startswith("unavailable") for value in checks.values())
