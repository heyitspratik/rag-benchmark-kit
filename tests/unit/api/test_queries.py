from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_bench.core.config import load_pipeline_config
from rag_bench.pipeline.indexer import Indexer

from .conftest import StubChatModel


@pytest.fixture
def indexed(pipeline_config: Path) -> Path:
    """Build the index once, so queries have something to search."""
    indexer = Indexer(load_pipeline_config(pipeline_config))
    try:
        indexer.build()
    finally:
        indexer.store.close()
    return pipeline_config


@pytest.fixture
def answering(indexed: Path, client: TestClient) -> TestClient:
    """A client whose provider is stubbed at the factory the generator really calls."""
    with (
        patch("rag_bench.pipeline.querier.check_llm_health"),
        patch("rag_bench.core.llm.build_chat_model", return_value=StubChatModel()),
    ):
        yield client


def test_a_question_returns_a_cited_answer(answering: TestClient) -> None:
    response = answering.post("/api/v1/queries", json={"question": "What fee may be charged?"})

    assert response.status_code == 200
    body = response.json()
    assert "reasonable fee" in body["answer"]
    assert body["abstained"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["section_refs"]


def test_a_query_creates_no_resource_so_it_is_200_not_201(answering: TestClient) -> None:
    response = answering.post("/api/v1/queries", json={"question": "What fee?"})

    assert response.status_code == 200
    assert "Location" not in response.headers


def test_contexts_are_withheld_unless_asked_for(answering: TestClient) -> None:
    # Passages are large; a caller opts in rather than paying for them by default.
    without = answering.post("/api/v1/queries", json={"question": "What fee?"}).json()
    with_them = answering.post(
        "/api/v1/queries", json={"question": "What fee?", "include_contexts": True}
    ).json()

    assert without["contexts"] == []
    assert with_them["contexts"]
    assert {"chunk_id", "text", "score", "rank"} <= set(with_them["contexts"][0])


def test_top_k_is_honoured(answering: TestClient) -> None:
    body = answering.post(
        "/api/v1/queries",
        json={"question": "What fee?", "top_k": 1, "include_contexts": True},
    ).json()

    assert len(body["contexts"]) == 1


def test_an_abstention_is_reported_as_such(answering: TestClient) -> None:
    body = answering.post("/api/v1/queries", json={"question": "An unanswerable question?"}).json()

    assert body["abstained"] is True
    assert body["citations"] == []


def test_timings_and_token_counts_are_returned(answering: TestClient) -> None:
    body = answering.post("/api/v1/queries", json={"question": "What fee?"}).json()

    assert body["retrieval_ms"] > 0
    assert body["prompt_tokens"] == 100
    assert body["completion_tokens"] == 12


def test_querying_before_an_index_exists_is_409(client: TestClient) -> None:
    with patch("rag_bench.pipeline.querier.check_llm_health"):
        response = client.post("/api/v1/queries", json={"question": "anything"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INDEX_NOT_READY"


def test_an_unknown_config_is_422(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/v1/queries",
        json={"question": "anything", "config": str(tmp_path / "absent.yaml")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CONFIG_INVALID"


def test_an_over_long_question_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/queries", json={"question": "x" * 5000})

    assert response.status_code == 422


def test_the_api_key_is_enforced_only_when_configured(app: FastAPI, indexed: Path) -> None:
    from pydantic import SecretStr

    app.state.settings = app.state.settings.model_copy(update={"api_key": SecretStr("s3cret")})
    with TestClient(app) as guarded:
        rejected = guarded.post("/api/v1/queries", json={"question": "What fee?"})
        with (
            patch("rag_bench.pipeline.querier.check_llm_health"),
            patch("rag_bench.core.llm.build_chat_model", return_value=StubChatModel()),
        ):
            accepted = guarded.post(
                "/api/v1/queries",
                json={"question": "What fee?"},
                headers={"X-API-Key": "s3cret"},
            )

    assert rejected.status_code == 401
    assert rejected.json()["error"]["code"] == "UNAUTHORISED"
    assert accepted.status_code == 200


def test_reads_stay_open_when_a_key_is_configured(app: FastAPI) -> None:
    # Listing components reveals nothing sensitive, and the quickstart should keep working.
    from pydantic import SecretStr

    app.state.settings = app.state.settings.model_copy(update={"api_key": SecretStr("s3cret")})
    with TestClient(app) as guarded:
        assert guarded.get("/api/v1/configurations").status_code == 200
        assert guarded.get("/health/live").status_code == 200
