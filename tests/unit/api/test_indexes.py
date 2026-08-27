from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient


def test_starting_a_build_is_accepted_not_completed(client: TestClient) -> None:
    # Ingesting a real corpus takes minutes, far longer than a request should hold open.
    response = client.post("/api/v1/indexes", json={})

    assert response.status_code == 202
    assert response.headers["Location"].startswith("/api/v1/indexes/")


def test_the_location_header_points_at_the_status_resource(client: TestClient) -> None:
    created = client.post("/api/v1/indexes", json={})

    followed = client.get(created.headers["Location"])

    assert followed.status_code == 200
    assert followed.json()["id"] == created.json()["id"]


def test_a_build_reaches_completion_and_reports_what_it_wrote(client: TestClient) -> None:
    # TestClient runs background tasks before returning, so the build is already done.
    created = client.post("/api/v1/indexes", json={})

    body = client.get(f"/api/v1/indexes/{created.json()['id']}").json()

    assert body["status"] == "completed"
    assert body["documents"] == 1
    assert body["chunks"] > 0
    assert body["dimension"] == 4
    assert body["finished_at"] is not None


def test_an_invalid_config_is_refused_before_being_accepted(
    client: TestClient, tmp_path: Path
) -> None:
    # A typo should fail with 422 here, not be accepted and fail invisibly later.
    response = client.post("/api/v1/indexes", json={"config": str(tmp_path / "absent.yaml")})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CONFIG_INVALID"


def test_an_unknown_build_is_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/indexes/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_a_malformed_id_is_422(client: TestClient) -> None:
    response = client.get("/api/v1/indexes/not-a-uuid")

    assert response.status_code == 422


def test_a_failing_build_is_recorded_rather_than_lost(
    client: TestClient, pipeline_config: Path
) -> None:
    # The request already returned 202, so the only place the failure can surface is
    # the status resource.
    broken = pipeline_config.read_text().replace("markdown_docs", "does_not_exist")
    pipeline_config.write_text(broken)

    created = client.post("/api/v1/indexes", json={})
    body = client.get(f"/api/v1/indexes/{created.json()['id']}").json()

    assert body["status"] == "failed"
    assert "UNKNOWN_COMPONENT" in body["error"]
