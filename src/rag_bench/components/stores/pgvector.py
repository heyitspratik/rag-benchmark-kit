"""Vector store backed by pgvector in the same Postgres that holds benchmark results.

This exists to keep the store interface honest. Qdrant and Postgres have almost nothing
in common operationally, so if both satisfy :class:`BaseVectorStore` without the pipeline
noticing, the abstraction is real rather than a Qdrant client in a thin disguise.

The collection table is created at runtime rather than by a migration, because a
``vector`` column must declare its dimension and the dimension is not known until an
embedder is chosen. A collection here is the same kind of object as a Qdrant collection:
runtime state owned by the store, not application schema. The ``vector`` extension itself
is schema, and does live in a migration.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence

from sqlalchemy import Engine, RowMapping, text
from sqlalchemy.exc import SQLAlchemyError

from rag_bench.core.exceptions import (
    ConfigValidationError,
    IndexNotReadyError,
    VectorStoreError,
)
from rag_bench.core.interfaces import BaseVectorStore, Vector
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Chunk, ScoredChunk
from rag_bench.core.registry import register_store
from rag_bench.db.session import build_engine

logger = get_logger(__name__)

# Collection names reach SQL as identifiers, which cannot be parameterised. Restricting
# them to this shape is what keeps a config file from becoming an injection vector.
_SAFE_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

_DISTANCE_OPERATORS = {"cosine": "<=>", "l2": "<->", "dot": "<#>"}

_ROWS_PER_FETCH = 256


@register_store("pgvector")
class PgVectorStore(BaseVectorStore):
    """Stores chunks and their vectors in a Postgres table managed by pgvector."""

    def __init__(
        self,
        collection: str = "rag_bench_chunks",
        distance: str = "cosine",
        batch_size: int = 128,
        engine: Engine | None = None,
    ) -> None:
        """Initialise the store.

        Args:
            collection: Table name; the experiment config owns this.
            distance: One of ``cosine``, ``l2``, ``dot``.
            batch_size: Rows per insert statement.
            engine: An existing engine, mainly for tests. One is built from the
                environment when omitted.

        Raises:
            ConfigValidationError: If the name, distance or batch size is invalid.
        """
        if not _SAFE_IDENTIFIER.match(collection):
            raise ConfigValidationError(
                f"store.params.collection must be a lowercase identifier, got {collection!r}",
                details={"collection": collection},
            )
        if distance not in _DISTANCE_OPERATORS:
            raise ConfigValidationError(
                "store.params.distance must be one of "
                f"{', '.join(sorted(_DISTANCE_OPERATORS))}, got {distance!r}"
            )
        if batch_size <= 0:
            raise ConfigValidationError(
                f"store.params.batch_size must be positive, got {batch_size}"
            )

        self._collection = collection
        self._operator = _DISTANCE_OPERATORS[distance]
        self._batch_size = batch_size
        self._engine = engine or build_engine()
        self._owns_engine = engine is None

    @property
    def collection(self) -> str:
        """Name of the table this store reads and writes."""
        return self._collection

    def ensure_collection(self, dimension: int) -> None:
        """Create the table and its index if absent, sized for the given vectors.

        Args:
            dimension: Vector length the table must accept.

        Raises:
            VectorStoreError: If Postgres is unreachable or rejects the DDL.
        """
        create = text(
            f"CREATE TABLE IF NOT EXISTS {self._collection} ("
            "  id UUID PRIMARY KEY,"
            "  doc_id TEXT NOT NULL,"
            "  ordinal INTEGER NOT NULL,"
            "  text TEXT NOT NULL,"
            "  char_start INTEGER NOT NULL,"
            "  char_end INTEGER NOT NULL,"
            "  section_refs JSONB NOT NULL DEFAULT '[]'::jsonb,"
            "  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,"
            f"  embedding vector({int(dimension)}) NOT NULL"
            ")"
        )
        index = text(
            f"CREATE INDEX IF NOT EXISTS {self._collection}_embedding_idx "
            f"ON {self._collection} USING hnsw (embedding vector_cosine_ops)"
        )
        try:
            with self._engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                connection.execute(create)
                connection.execute(index)
        except SQLAlchemyError as exc:
            raise VectorStoreError(
                f"Could not create pgvector collection {self._collection!r}: {exc}",
                details={"collection": self._collection},
            ) from exc
        logger.info("store.collection_created", collection=self._collection, dimension=dimension)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        """Insert or replace chunks and their vectors.

        Args:
            chunks: The chunks to store.
            vectors: Vectors positionally aligned with ``chunks``.

        Raises:
            ValueError: If the two sequences differ in length.
            VectorStoreError: If Postgres rejects the write.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"upsert needs one vector per chunk, got {len(chunks)} chunks "
                f"and {len(vectors)} vectors"
            )
        if not chunks:
            return

        statement = text(
            f"INSERT INTO {self._collection} "
            "(id, doc_id, ordinal, text, char_start, char_end, section_refs, metadata, embedding) "
            "VALUES (:id, :doc_id, :ordinal, :text, :char_start, :char_end, "
            "CAST(:section_refs AS jsonb), CAST(:metadata AS jsonb), CAST(:embedding AS vector)) "
            "ON CONFLICT (id) DO UPDATE SET "
            "doc_id = EXCLUDED.doc_id, ordinal = EXCLUDED.ordinal, text = EXCLUDED.text, "
            "char_start = EXCLUDED.char_start, char_end = EXCLUDED.char_end, "
            "section_refs = EXCLUDED.section_refs, metadata = EXCLUDED.metadata, "
            "embedding = EXCLUDED.embedding"
        )
        rows = [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "ordinal": chunk.ordinal,
                "text": chunk.text,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "section_refs": json.dumps(list(chunk.section_refs)),
                "metadata": json.dumps(dict(chunk.metadata)),
                "embedding": _as_vector_literal(vector),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            with self._engine.begin() as connection:
                for start in range(0, len(rows), self._batch_size):
                    connection.execute(statement, rows[start : start + self._batch_size])
        except SQLAlchemyError as exc:
            raise VectorStoreError(
                f"Could not write to pgvector collection {self._collection!r}: {exc}",
                details={"collection": self._collection, "rows": len(rows)},
            ) from exc

    def search(self, vector: Vector, k: int) -> list[ScoredChunk]:
        """Return the ``k`` nearest chunks to a query vector, best first.

        Args:
            vector: The query vector.
            k: Maximum number of results.

        Returns:
            Scored chunks ordered by descending similarity, so the score means the same
            thing here as it does for the Qdrant backend.

        Raises:
            IndexNotReadyError: If the collection does not exist.
            VectorStoreError: If Postgres rejects the query.
        """
        self._require_collection()
        statement = text(
            "SELECT id, doc_id, ordinal, text, char_start, char_end, section_refs, metadata, "
            f"1 - (embedding {self._operator} CAST(:query AS vector)) AS score "
            f"FROM {self._collection} "
            f"ORDER BY embedding {self._operator} CAST(:query AS vector) "
            "LIMIT :limit"
        )
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(
                    statement, {"query": _as_vector_literal(vector), "limit": k}
                ).mappings()
                return [
                    ScoredChunk(chunk=_to_chunk(row), score=float(row["score"]), rank=rank)
                    for rank, row in enumerate(rows)
                ]
        except SQLAlchemyError as exc:
            raise VectorStoreError(
                f"pgvector search failed on {self._collection!r}: {exc}",
                details={"collection": self._collection},
            ) from exc

    def iter_chunks(self) -> Iterator[Chunk]:
        """Stream every stored chunk.

        Yields:
            Each stored chunk. Vectors are left behind, since the lexical retriever that
            needs this only wants the text.

        Raises:
            IndexNotReadyError: If the collection does not exist.
            VectorStoreError: If Postgres rejects the read.
        """
        self._require_collection()
        statement = text(
            "SELECT id, doc_id, ordinal, text, char_start, char_end, section_refs, metadata "
            f"FROM {self._collection} ORDER BY doc_id, ordinal"
        )
        try:
            with self._engine.connect() as connection:
                result = connection.execution_options(stream_results=True).execute(statement)
                for row in result.mappings().yield_per(_ROWS_PER_FETCH):
                    yield _to_chunk(row)
        except SQLAlchemyError as exc:
            raise VectorStoreError(
                f"pgvector scan failed on {self._collection!r}: {exc}",
                details={"collection": self._collection},
            ) from exc

    def count(self) -> int:
        """Number of chunks stored, or zero when the collection is absent."""
        if not self._collection_exists():
            return 0
        try:
            with self._engine.connect() as connection:
                total = connection.execute(
                    text(f"SELECT count(*) FROM {self._collection}")
                ).scalar_one()
        except SQLAlchemyError as exc:
            raise VectorStoreError(
                f"Could not count pgvector collection {self._collection!r}: {exc}",
                details={"collection": self._collection},
            ) from exc
        return int(total)

    def delete_collection(self) -> None:
        """Drop the table and everything in it, if it exists."""
        try:
            with self._engine.begin() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS {self._collection}"))
        except SQLAlchemyError as exc:
            raise VectorStoreError(
                f"Could not drop pgvector collection {self._collection!r}: {exc}",
                details={"collection": self._collection},
            ) from exc
        logger.info("store.collection_deleted", collection=self._collection)

    def close(self) -> None:
        """Dispose the engine, unless it was handed in by the caller."""
        if self._owns_engine:
            self._engine.dispose()

    def _collection_exists(self) -> bool:
        """Whether the collection table is present."""
        try:
            with self._engine.connect() as connection:
                return bool(
                    connection.execute(
                        text("SELECT to_regclass(:name) IS NOT NULL"),
                        {"name": self._collection},
                    ).scalar_one()
                )
        except SQLAlchemyError as exc:
            raise VectorStoreError(
                f"Could not reach Postgres for collection {self._collection!r}: {exc}",
                details={"collection": self._collection},
            ) from exc

    def _require_collection(self) -> None:
        """Fail with a readable message when the index has not been built yet."""
        if not self._collection_exists():
            raise IndexNotReadyError(
                f"pgvector collection {self._collection!r} does not exist. "
                "Run `rag-bench index build` first.",
                details={"collection": self._collection},
            )


def _as_vector_literal(vector: Vector) -> str:
    """Render a vector in the bracketed form pgvector parses."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def _to_chunk(row: RowMapping) -> Chunk:
    """Rebuild a chunk from one result row.

    Args:
        row: A mapping-style row from a ``SELECT`` over the collection table.

    Returns:
        The reconstructed chunk.

    Raises:
        VectorStoreError: If the row does not validate, which means the table was written
            by something other than this store.
    """
    try:
        return Chunk.model_validate({**row, "id": str(row["id"])})
    except (KeyError, ValueError) as exc:
        raise VectorStoreError(f"pgvector row is not a valid chunk: {exc}") from exc
