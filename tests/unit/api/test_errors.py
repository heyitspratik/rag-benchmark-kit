from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_bench.core.exceptions import IndexNotReadyError

_ENVELOPE_KEYS = {"code", "message", "details", "request_id"}


def test_a_project_error_uses_its_own_code_and_status(client: TestClient) -> None:
    with patch("rag_bench.api.services.Querier", side_effect=IndexNotReadyError("build it first")):
        response = client.post("/api/v1/queries", json={"question": "anything"})

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "INDEX_NOT_READY"
    assert error["message"] == "build it first"


def test_every_error_shares_one_envelope(client: TestClient) -> None:
    # A client writes the error handling once, not once per resource.
    responses = [
        client.post("/api/v1/queries", json={}),
        client.get("/api/v1/benchmark-runs/not-a-uuid"),
        client.get("/api/v1/does-not-exist"),
    ]

    for response in responses:
        assert set(response.json()["error"]) == _ENVELOPE_KEYS


def test_a_malformed_body_is_422_and_names_the_field(client: TestClient) -> None:
    response = client.post("/api/v1/queries", json={"question": ""})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert any("question" in item["field"] for item in error["details"]["errors"])


def test_an_unknown_field_is_rejected_rather_than_ignored(client: TestClient) -> None:
    response = client.post("/api/v1/queries", json={"question": "a question", "temperture": 0.5})

    assert response.status_code == 422


def test_an_unknown_path_still_uses_the_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "HTTP_ERROR"


def test_an_unexpected_error_does_not_leak_internals(forgiving_client: TestClient) -> None:
    # The message may carry internals, so it is replaced by the request ID.
    with patch(
        "rag_bench.api.services.ConfigurationService.components",
        side_effect=RuntimeError("connection string postgres://user:hunter2@host/db"),
    ):
        response = forgiving_client.get("/api/v1/configurations")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "INTERNAL_ERROR"
    assert "hunter2" not in error["message"]
    assert error["request_id"]


def test_the_error_envelope_carries_the_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/nope", headers={"X-Request-ID": "trace-me"})

    assert response.json()["error"]["request_id"] == "trace-me"
    assert response.headers["X-Request-ID"] == "trace-me"
