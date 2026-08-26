"""Checks that need no database.

The behaviour against real Postgres lives in tests/integration/, because a fake would
only prove the fake works.
"""

import pytest
from sqlalchemy import create_engine

from rag_bench.components.stores.pgvector import PgVectorStore, _as_vector_literal
from rag_bench.core.exceptions import ConfigValidationError


def test_a_vector_is_rendered_in_the_form_pgvector_parses() -> None:
    assert _as_vector_literal([1.0, -0.5, 0.25]) == "[1.0,-0.5,0.25]"


def test_an_empty_vector_still_renders() -> None:
    assert _as_vector_literal([]) == "[]"


@pytest.mark.parametrize(
    "collection",
    [
        "chunks; DROP TABLE benchmark_runs",
        "chunks--",
        'chunks"',
        "Chunks",
        "1chunks",
        "",
        "x" * 64,
    ],
)
def test_an_unsafe_collection_name_is_refused(collection: str) -> None:
    # Table names cannot be parameterised, so this check is what keeps a config file
    # from becoming an injection vector.
    with pytest.raises(ConfigValidationError, match="lowercase identifier"):
        PgVectorStore(collection=collection)


@pytest.mark.parametrize("collection", ["chunks", "rag_bench_chunks", "_private", "a1_b2"])
def test_a_safe_collection_name_is_accepted(collection: str) -> None:
    store = PgVectorStore(collection=collection)

    assert store.collection == collection
    store.close()


def test_an_unknown_distance_is_rejected() -> None:
    with pytest.raises(ConfigValidationError, match="distance must be one of"):
        PgVectorStore(distance="manhattan")


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_non_positive_batch_size_is_rejected(batch_size: int) -> None:
    with pytest.raises(ConfigValidationError, match="batch_size must be positive"):
        PgVectorStore(batch_size=batch_size)


def test_a_supplied_engine_is_not_disposed_by_close() -> None:
    engine = create_engine("sqlite://")
    store = PgVectorStore(engine=engine)

    store.close()

    # Still usable, because the caller owns the engine it handed in. The benchmark
    # shares one engine across configurations, so a store closing it would break the
    # next configuration in the sweep.
    with engine.connect() as connection:
        assert connection is not None
    engine.dispose()
