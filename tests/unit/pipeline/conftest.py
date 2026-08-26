"""A markdown corpus and an offline pipeline config for indexer tests."""

from pathlib import Path

import pytest

from rag_bench.core.config import ComponentConfig, CorpusConfig, PipelineConfig
from tests.conftest import OFFLINE_DIMENSION, OFFLINE_EMBEDDER

DIMENSION = OFFLINE_DIMENSION

_PAGE = """# Controllers

Some introductory prose about controllers and their duties.

## Fees

A controller may charge a reasonable fee based on administrative costs.

## Refusals

A controller may refuse to act on a manifestly unfounded request.
"""


@pytest.fixture
def markdown_corpus(tmp_path: Path) -> Path:
    """A one-page Markdown corpus whose headings become sections."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "controllers.md").write_text(_PAGE)
    return root


@pytest.fixture
def config(markdown_corpus: Path, tmp_path: Path) -> PipelineConfig:
    """A pipeline wired entirely to offline components."""
    return PipelineConfig(
        corpus=CorpusConfig(name="markdown_docs", path=markdown_corpus),
        chunker=ComponentConfig(name="structural", params={"max_chars": 200, "overlap": 0}),
        embedder=ComponentConfig(name=OFFLINE_EMBEDDER),
        store=ComponentConfig(
            name="qdrant",
            params={"collection": "indexer_test", "url": f"file:{tmp_path / 'qdrant'}"},
        ),
        retriever=ComponentConfig(name="dense"),
        generator=ComponentConfig(name="cited"),
    )
