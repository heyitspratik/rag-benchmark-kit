"""A complete offline benchmark stack: local corpus, local store, stubbed provider."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import Engine, create_engine

from rag_bench.core.config import SweepConfig
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

_QUESTIONS = [
    (
        '{"id": "q1", "question": "What fee may a controller charge?", "ground_truth": "A '
        'reasonable fee.", "source_refs": ["controllers.md#Fees"], "difficulty": "single_hop"}'
    ),
    (
        '{"id": "q2", "question": "When may a controller refuse a request?", "ground_truth": '
        '"When manifestly unfounded.", "source_refs": ["controllers.md#Refusals"], '
        '"difficulty": "multi_hop"}'
    ),
    (
        '{"id": "q3", "question": "An unanswerable question about tax law?", "ground_truth": '
        '"Not in the corpus.", "difficulty": "negative"}'
    ),
]


class StubChatModel:
    """Cites the first passage, and declines on the deliberately unanswerable question."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list[tuple[str, str]]) -> AIMessage:
        self.calls += 1
        if DECLINE_MARKER in messages[1][1]:
            return AIMessage(content="INSUFFICIENT_CONTEXT")
        return AIMessage(
            content="An answer grounded in the context [1].",
            usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        )


@pytest.fixture
def bench_engine() -> Iterator[Engine]:
    """A throwaway database. Alembic owns the real schema; this mirrors it for tests."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A corpus, an eval set and a base config, all local and free to run."""
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "controllers.md").write_text(_PAGE)

    (tmp_path / "eval.jsonl").write_text("\n".join(_QUESTIONS) + "\n")

    (tmp_path / "base.yaml").write_text(
        f"corpus:\n  name: markdown_docs\n  path: {corpus}\n"
        f"chunker:\n  name: structural\n  params: {{max_chars: 200, overlap: 0}}\n"
        f"embedder:\n  name: {OFFLINE_EMBEDDER}\n"
        f"store:\n"
        f"  name: qdrant\n"
        f"  params: {{collection: bench_test, url: 'file:{tmp_path / 'qdrant'}'}}\n"
        f"retriever:\n  name: dense\n  params: {{top_k: 3}}\n"
        f"generator:\n  name: cited\n"
    )
    return tmp_path


@pytest.fixture
def sweep(workspace: Path) -> SweepConfig:
    """A two-by-two grid over the offline workspace."""
    return SweepConfig(
        name="offline_grid",
        eval_set=workspace / "eval.jsonl",
        base_config=workspace / "base.yaml",
        sweep={"chunker": ["fixed", "structural"], "retriever": ["dense", "hybrid"]},
        params={
            "chunker": {
                "fixed": {"max_chars": 200, "overlap": 0},
                "structural": {"max_chars": 200, "overlap": 0},
            },
            "retriever": {
                "dense": {"top_k": 3},
                "hybrid": {"top_k": 3, "bm25_weight": 0.4, "dense_weight": 0.6},
            },
        },
    )


@pytest.fixture
def chat_model() -> StubChatModel:
    return StubChatModel()
