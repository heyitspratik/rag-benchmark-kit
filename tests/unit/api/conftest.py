"""An application wired to offline components and a throwaway database."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from rag_bench.api.main import create_app
from rag_bench.core.settings import Settings
from rag_bench.db.models import Base
from tests.conftest import OFFLINE_EMBEDDER

DECLINE_MARKER = "unanswerable"

_PAGE = """# Controllers

Introductory prose about controllers and their duties under the regulation.

## Fees

A controller may charge a reasonable fee based on administrative costs for further copies.

## Refusals

A controller may refuse to act on a manifestly unfounded or excessive request.
"""


class StubChatModel:
    """Stands in for a provider, so no test needs a key or a running Ollama."""

    def invoke(self, messages: list[tuple[str, str]]) -> AIMessage:
        if DECLINE_MARKER in messages[1][1]:
            return AIMessage(content="INSUFFICIENT_CONTEXT")
        return AIMessage(
            content="A controller may charge a reasonable fee [1].",
            usage_metadata={"input_tokens": 100, "output_tokens": 12, "total_tokens": 112},
        )


@pytest.fixture
def api_engine() -> Iterator[Engine]:
    # TestClient runs the application in its own thread, and an in-memory SQLite
    # connection belongs to the thread that opened it. A static pool shares the one
    # connection so the test and the app see the same database.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def pipeline_config(tmp_path: Path) -> Path:
    """A config wired entirely to local components."""
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "controllers.md").write_text(_PAGE)

    config = tmp_path / "api.yaml"
    config.write_text(
        f"corpus:\n  name: markdown_docs\n  path: {corpus}\n"
        f"chunker:\n  name: structural\n  params: {{max_chars: 200, overlap: 0}}\n"
        f"embedder:\n  name: {OFFLINE_EMBEDDER}\n"
        f"store:\n"
        f"  name: qdrant\n"
        f"  params: {{collection: api_test, url: 'file:{tmp_path / 'qdrant'}'}}\n"
        f"retriever:\n  name: dense\n  params: {{top_k: 3}}\n"
        f"generator:\n  name: cited\n"
    )
    return config


@pytest.fixture
def app(api_engine: Engine, pipeline_config: Path) -> FastAPI:
    return create_app(
        settings=Settings(_env_file=None),
        engine=api_engine,
        default_config=pipeline_config,
    )


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as open_client:
        yield open_client


@pytest.fixture
def forgiving_client(app: FastAPI) -> Iterator[TestClient]:
    """A client that returns a 500 rather than re-raising, as a real caller would see."""
    with TestClient(app, raise_server_exceptions=False) as open_client:
        yield open_client
