from fastapi.testclient import TestClient


def test_a_request_id_is_generated_when_none_is_given(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.headers["X-Request-ID"]


def test_a_caller_supplied_request_id_is_propagated(client: TestClient) -> None:
    # The caller's ID is what lets them find their request in the server's logs.
    response = client.get("/health/live", headers={"X-Request-ID": "abc-123"})

    assert response.headers["X-Request-ID"] == "abc-123"


def test_each_request_gets_a_distinct_id(client: TestClient) -> None:
    first = client.get("/health/live").headers["X-Request-ID"]
    second = client.get("/health/live").headers["X-Request-ID"]

    assert first != second


def test_the_request_id_is_bound_to_the_logs(client: TestClient, caplog: "object") -> None:
    import logging

    from rag_bench.core.logging import configure_logging

    configure_logging(app_env="prod")
    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        client.get("/api/v1/configurations", headers={"X-Request-ID": "bound-id"})

    assert "bound-id" in caplog.text  # type: ignore[attr-defined]
