"""The OpenAPI document is the API's documentation, so it is checked rather than assumed."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

EXPECTED_OPERATIONS = {
    ("post", "/api/v1/queries"),
    ("get", "/api/v1/configurations"),
    ("post", "/api/v1/indexes"),
    ("get", "/api/v1/indexes/{index_id}"),
    ("get", "/api/v1/benchmark-runs"),
    ("get", "/api/v1/benchmark-runs/{run_id}"),
    ("get", "/health/live"),
    ("get", "/health/ready"),
}


@pytest.fixture
def spec(client: TestClient) -> dict[str, Any]:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def test_every_documented_endpoint_exists(spec: dict[str, Any]) -> None:
    actual = {(method, path) for path, operations in spec["paths"].items() for method in operations}

    assert actual == EXPECTED_OPERATIONS


def test_every_operation_has_a_summary(spec: dict[str, Any]) -> None:
    missing = [
        f"{method} {path}"
        for path, operations in spec["paths"].items()
        for method, operation in operations.items()
        if not operation.get("summary")
    ]

    assert missing == []


def test_no_operation_returns_an_undeclared_body(spec: dict[str, Any]) -> None:
    # A route returning a bare dict would leave the schema as a free-form object, and
    # the document would describe the API only in outline.
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            success = next((code for code in operation["responses"] if code.startswith("2")), None)
            assert success is not None, f"{method} {path} declares no success response"
            content = operation["responses"][success].get("content", {})
            schema = content.get("application/json", {}).get("schema", {})
            assert schema, f"{method} {path} declares no response schema"


def test_paths_use_plural_nouns_and_no_verbs(spec: dict[str, Any]) -> None:
    verbs = ("create", "build", "run", "get", "list", "submit", "trigger")
    for path in spec["paths"]:
        segments = [s for s in path.split("/") if s and not s.startswith("{")]
        assert not any(segment in verbs for segment in segments), path


def test_error_responses_are_documented_with_the_shared_envelope(
    spec: dict[str, Any],
) -> None:
    query = spec["paths"]["/api/v1/queries"]["post"]["responses"]

    assert {"409", "422", "502"} <= set(query)
    for code in ("409", "422", "502"):
        ref = query[code]["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith("ErrorResponse")


def test_the_creation_endpoint_documents_202(spec: dict[str, Any]) -> None:
    # Index builds take minutes, so they are accepted rather than completed inline.
    assert "202" in spec["paths"]["/api/v1/indexes"]["post"]["responses"]


def test_the_documentation_pages_render(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
