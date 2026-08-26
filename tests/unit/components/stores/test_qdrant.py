"""Exercises the real Qdrant client in its embedded mode, so no server is needed."""

from collections.abc import Iterator

import pytest

from rag_bench.components.stores.qdrant import MEMORY_URL, QdrantStore
from rag_bench.core.exceptions import (
    ConfigValidationError,
    IndexNotReadyError,
    VectorStoreError,
)
from rag_bench.core.models import Chunk

DIMENSION = 3


def _chunk(ordinal: int, text: str, refs: tuple[str, ...] = ()) -> Chunk:
    return Chunk.create(
        doc_id="gdpr",
        ordinal=ordinal,
        text=text,
        char_start=ordinal * 10,
        char_end=ordinal * 10 + len(text),
        metadata={"section_title": f"Article {ordinal}"},
    ).with_sections(refs)


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    instance = QdrantStore(collection="test_collection", url=MEMORY_URL)
    yield instance
    instance.close()


@pytest.fixture
def populated(store: QdrantStore) -> QdrantStore:
    store.ensure_collection(DIMENSION)
    store.upsert(
        [
            _chunk(0, "east vector", ("GDPR Art. 1",)),
            _chunk(1, "north vector", ("GDPR Art. 2",)),
            _chunk(2, "up vector"),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    return store


def test_search_ranks_by_similarity(populated: QdrantStore) -> None:
    results = populated.search([0.9, 0.1, 0.0], k=3)

    assert [r.chunk.text for r in results] == ["east vector", "north vector", "up vector"]
    assert [r.rank for r in results] == [0, 1, 2]


def test_search_respects_k(populated: QdrantStore) -> None:
    assert len(populated.search([1.0, 0.0, 0.0], k=2)) == 2


def test_a_chunk_survives_the_round_trip(populated: QdrantStore) -> None:
    best = populated.search([1.0, 0.0, 0.0], k=1)[0].chunk

    assert best.id == _chunk(0, "east vector").id
    assert best.doc_id == "gdpr"
    assert best.section_refs == ("GDPR Art. 1",)
    assert best.metadata == {"section_title": "Article 0"}
    assert (best.char_start, best.char_end) == (0, 11)


def test_iter_chunks_streams_everything(populated: QdrantStore) -> None:
    texts = {chunk.text for chunk in populated.iter_chunks()}

    assert texts == {"east vector", "north vector", "up vector"}


def test_count_reflects_what_was_written(populated: QdrantStore) -> None:
    assert populated.count() == 3


def test_upserting_the_same_chunk_twice_does_not_duplicate(populated: QdrantStore) -> None:
    populated.upsert([_chunk(0, "east vector", ("GDPR Art. 1",))], [[1.0, 0.0, 0.0]])

    assert populated.count() == 3


def test_ensure_collection_is_idempotent(store: QdrantStore) -> None:
    store.ensure_collection(DIMENSION)
    store.ensure_collection(DIMENSION)

    assert store.count() == 0


def test_count_is_zero_before_the_collection_exists(store: QdrantStore) -> None:
    assert store.count() == 0


def test_delete_collection_removes_everything(populated: QdrantStore) -> None:
    populated.delete_collection()

    assert populated.count() == 0


def test_deleting_an_absent_collection_is_a_no_op(store: QdrantStore) -> None:
    store.delete_collection()


def test_searching_before_the_index_exists_names_the_build_command(store: QdrantStore) -> None:
    with pytest.raises(IndexNotReadyError, match="rag-bench index build"):
        store.search([1.0, 0.0, 0.0], k=1)


def test_streaming_before_the_index_exists_is_reported(store: QdrantStore) -> None:
    with pytest.raises(IndexNotReadyError):
        list(store.iter_chunks())


def test_mismatched_chunk_and_vector_counts_are_rejected(store: QdrantStore) -> None:
    store.ensure_collection(DIMENSION)

    with pytest.raises(ValueError, match="one vector per chunk"):
        store.upsert([_chunk(0, "a"), _chunk(1, "b")], [[1.0, 0.0, 0.0]])


def test_upserting_nothing_is_a_no_op(store: QdrantStore) -> None:
    store.ensure_collection(DIMENSION)
    store.upsert([], [])

    assert store.count() == 0


def test_a_wrong_sized_vector_is_reported(store: QdrantStore) -> None:
    store.ensure_collection(DIMENSION)

    with pytest.raises(VectorStoreError, match="Could not write"):
        store.upsert([_chunk(0, "a")], [[1.0, 0.0]])


def test_batching_writes_every_point() -> None:
    store = QdrantStore(collection="batched", url=MEMORY_URL, batch_size=2)
    store.ensure_collection(DIMENSION)
    chunks = [_chunk(i, f"chunk {i}") for i in range(5)]

    store.upsert(chunks, [[1.0, 0.0, 0.0]] * 5)

    assert store.count() == 5
    store.close()


def test_an_unknown_distance_is_rejected() -> None:
    with pytest.raises(ConfigValidationError, match="distance must be one of"):
        QdrantStore(distance="manhattan", url=MEMORY_URL)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_non_positive_batch_size_is_rejected(batch_size: int) -> None:
    with pytest.raises(ConfigValidationError, match="batch_size must be positive"):
        QdrantStore(batch_size=batch_size, url=MEMORY_URL)
