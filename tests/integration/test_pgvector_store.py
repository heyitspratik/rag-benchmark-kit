from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from rag_bench.components.stores.pgvector import PgVectorStore
from rag_bench.core.exceptions import IndexNotReadyError
from rag_bench.core.models import Chunk

pytestmark = pytest.mark.integration

DIMENSION = 3
COLLECTION = "test_pgvector_chunks"


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
def store(postgres_engine: Engine) -> Iterator[PgVectorStore]:
    instance = PgVectorStore(collection=COLLECTION, engine=postgres_engine)
    instance.delete_collection()
    yield instance
    instance.delete_collection()


@pytest.fixture
def populated(store: PgVectorStore) -> PgVectorStore:
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


def test_search_ranks_by_similarity(populated: PgVectorStore) -> None:
    results = populated.search([0.9, 0.1, 0.0], k=3)

    assert [r.chunk.text for r in results] == ["east vector", "north vector", "up vector"]
    assert [r.rank for r in results] == [0, 1, 2]


def test_scores_increase_with_similarity(populated: PgVectorStore) -> None:
    # The interface promises a similarity, not a distance, so both backends agree.
    best, worst = (
        populated.search([1.0, 0.0, 0.0], k=3)[0],
        populated.search([1.0, 0.0, 0.0], k=3)[-1],
    )

    assert best.score > worst.score


def test_a_chunk_survives_the_round_trip(populated: PgVectorStore) -> None:
    best = populated.search([1.0, 0.0, 0.0], k=1)[0].chunk

    assert best.id == _chunk(0, "east vector").id
    assert best.section_refs == ("GDPR Art. 1",)
    assert best.metadata == {"section_title": "Article 0"}
    assert (best.char_start, best.char_end) == (0, 11)


def test_iter_chunks_streams_everything(populated: PgVectorStore) -> None:
    assert {chunk.text for chunk in populated.iter_chunks()} == {
        "east vector",
        "north vector",
        "up vector",
    }


def test_upserting_the_same_chunk_twice_does_not_duplicate(populated: PgVectorStore) -> None:
    populated.upsert([_chunk(0, "east vector", ("GDPR Art. 1",))], [[1.0, 0.0, 0.0]])

    assert populated.count() == 3


def test_ensure_collection_is_idempotent(store: PgVectorStore) -> None:
    store.ensure_collection(DIMENSION)
    store.ensure_collection(DIMENSION)

    assert store.count() == 0


def test_count_is_zero_before_the_collection_exists(store: PgVectorStore) -> None:
    assert store.count() == 0


def test_searching_before_the_index_exists_names_the_build_command(
    store: PgVectorStore,
) -> None:
    with pytest.raises(IndexNotReadyError, match="rag-bench index build"):
        store.search([1.0, 0.0, 0.0], k=1)


def test_mismatched_chunk_and_vector_counts_are_rejected(store: PgVectorStore) -> None:
    store.ensure_collection(DIMENSION)

    with pytest.raises(ValueError, match="one vector per chunk"):
        store.upsert([_chunk(0, "a"), _chunk(1, "b")], [[1.0, 0.0, 0.0]])


def test_deleting_the_collection_removes_everything(populated: PgVectorStore) -> None:
    populated.delete_collection()

    assert populated.count() == 0
