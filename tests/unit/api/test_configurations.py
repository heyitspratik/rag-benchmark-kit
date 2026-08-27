from fastapi.testclient import TestClient


def test_every_pipeline_stage_is_listed(client: TestClient) -> None:
    response = client.get("/api/v1/configurations")

    assert response.status_code == 200
    components = response.json()["components"]
    assert set(components) == {
        "loader",
        "chunker",
        "embedder",
        "store",
        "retriever",
        "generator",
    }


def test_the_registered_implementations_are_reported(client: TestClient) -> None:
    # Read from the registries, so a newly registered component appears without this
    # route or its schema changing.
    components = client.get("/api/v1/configurations").json()["components"]

    assert {"fixed", "recursive", "semantic", "structural"} <= set(components["chunker"])
    assert {"dense", "hybrid", "hybrid_rerank"} <= set(components["retriever"])
    assert {"qdrant", "pgvector"} <= set(components["store"])


def test_names_are_sorted(client: TestClient) -> None:
    components = client.get("/api/v1/configurations").json()["components"]

    assert all(names == sorted(names) for names in components.values())
