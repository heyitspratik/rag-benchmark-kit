"""Qdrant vector store, the default backend.

Connection details come from the environment and the collection name from the experiment
YAML, which keeps a committed config portable between a laptop and CI without editing it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from pydantic import ValidationError
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag_bench.core.exceptions import (
    ConfigValidationError,
    IndexNotReadyError,
    VectorStoreError,
)
from rag_bench.core.interfaces import BaseVectorStore, Vector
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Chunk, ScoredChunk
from rag_bench.core.registry import register_store
from rag_bench.core.settings import get_settings

if TYPE_CHECKING:
    from qdrant_client.conversions.common_types import PointId, ScoredPoint

logger = get_logger(__name__)

_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}

#: Points fetched per scroll page when streaming the collection back out.
_SCROLL_PAGE = 256

#: URL that selects qdrant-client's embedded mode instead of a server.
MEMORY_URL = ":memory:"

#: URL prefix that selects embedded mode backed by a local directory.
FILE_URL_PREFIX = "file:"


def _connect(url: str, api_key: str | None, timeout_s: float) -> QdrantClient:
    """Open a Qdrant client, allowing the embedded backend as well as a server.

    Embedded mode is what lets the test suite and a first local run exercise the real
    store without Docker; the server is still the default and what compose brings up.

    File-backed embedded mode takes an exclusive lock on its directory, so only one
    process can use a given path at a time. Two benchmark runs sharing one directory is
    a server-mode job.
    """
    if url == MEMORY_URL:
        return QdrantClient(location=MEMORY_URL)
    if url.startswith(FILE_URL_PREFIX):
        return QdrantClient(path=url.removeprefix(FILE_URL_PREFIX))
    return QdrantClient(url=url, api_key=api_key, timeout=int(timeout_s))


@register_store("qdrant")
class QdrantStore(BaseVectorStore):
    """Stores chunks and their vectors in a Qdrant collection."""

    def __init__(
        self,
        collection: str = "rag_bench",
        distance: str = "cosine",
        batch_size: int = 128,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialise the store.

        Args:
            collection: Collection name; the experiment config owns this.
            distance: One of ``cosine``, ``dot``, ``euclid``.
            batch_size: Points per upsert request.
            url: Override the URL from the environment, mainly for tests.
            api_key: Override the API key from the environment.

        Raises:
            ConfigValidationError: If the distance or batch size is invalid.
        """
        if distance not in _DISTANCES:
            raise ConfigValidationError(
                f"store.params.distance must be one of {', '.join(sorted(_DISTANCES))}, "
                f"got {distance!r}"
            )
        if batch_size <= 0:
            raise ConfigValidationError(
                f"store.params.batch_size must be positive, got {batch_size}"
            )

        settings = get_settings().qdrant
        secret = settings.api_key.get_secret_value() if settings.api_key else None
        self._collection = collection
        self._distance = _DISTANCES[distance]
        self._batch_size = batch_size
        self._client = _connect(url or settings.url, api_key or secret, settings.timeout_s)

    @property
    def collection(self) -> str:
        """Name of the collection this store reads and writes."""
        return self._collection

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it is absent, sized for the given vectors.

        Args:
            dimension: Vector length the collection must accept.

        Raises:
            VectorStoreError: If Qdrant is unreachable or rejects the request.
        """
        try:
            if self._client.collection_exists(self._collection):
                return
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=dimension, distance=self._distance),
            )
        except (UnexpectedResponse, OSError, ValueError) as exc:
            raise VectorStoreError(
                f"Could not create Qdrant collection {self._collection!r}: {exc}",
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
            VectorStoreError: If Qdrant rejects the write.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"upsert needs one vector per chunk, got {len(chunks)} chunks "
                f"and {len(vectors)} vectors"
            )
        if not chunks:
            return

        points = [
            models.PointStruct(id=chunk.id, vector=list(vector), payload=_to_payload(chunk))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            for start in range(0, len(points), self._batch_size):
                self._client.upsert(
                    collection_name=self._collection,
                    points=points[start : start + self._batch_size],
                    wait=True,
                )
        except (UnexpectedResponse, OSError, ValueError) as exc:
            raise VectorStoreError(
                f"Could not write to Qdrant collection {self._collection!r}: {exc}",
                details={"collection": self._collection, "points": len(points)},
            ) from exc

    def search(self, vector: Vector, k: int) -> list[ScoredChunk]:
        """Return the ``k`` nearest chunks to a query vector, best first.

        Args:
            vector: The query vector.
            k: Maximum number of results.

        Returns:
            Scored chunks ordered by descending similarity.

        Raises:
            IndexNotReadyError: If the collection does not exist.
            VectorStoreError: If Qdrant rejects the query.
        """
        self._require_collection()
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                limit=k,
                with_payload=True,
            )
        except (UnexpectedResponse, OSError, ValueError) as exc:
            raise VectorStoreError(
                f"Qdrant search failed on {self._collection!r}: {exc}",
                details={"collection": self._collection},
            ) from exc
        return [_to_scored_chunk(point, rank) for rank, point in enumerate(response.points)]

    def iter_chunks(self) -> Iterator[Chunk]:
        """Stream every stored chunk, a page at a time.

        Yields:
            Each stored chunk.

        Raises:
            IndexNotReadyError: If the collection does not exist.
            VectorStoreError: If Qdrant rejects the scroll.
        """
        self._require_collection()
        offset: PointId | None = None
        while True:
            try:
                points, offset = self._client.scroll(
                    collection_name=self._collection,
                    limit=_SCROLL_PAGE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except (UnexpectedResponse, OSError, ValueError) as exc:
                raise VectorStoreError(
                    f"Qdrant scroll failed on {self._collection!r}: {exc}",
                    details={"collection": self._collection},
                ) from exc
            for point in points:
                yield _from_payload(point.id, point.payload)
            if offset is None:
                return

    def count(self) -> int:
        """Number of chunks stored, or zero when the collection is absent."""
        try:
            if not self._client.collection_exists(self._collection):
                return 0
            return int(self._client.count(self._collection, exact=True).count)
        except (UnexpectedResponse, OSError, ValueError) as exc:
            raise VectorStoreError(
                f"Could not count Qdrant collection {self._collection!r}: {exc}",
                details={"collection": self._collection},
            ) from exc

    def delete_collection(self) -> None:
        """Drop the collection and everything in it, if it exists."""
        try:
            if self._client.collection_exists(self._collection):
                self._client.delete_collection(self._collection)
                logger.info("store.collection_deleted", collection=self._collection)
        except (UnexpectedResponse, OSError, ValueError) as exc:
            raise VectorStoreError(
                f"Could not delete Qdrant collection {self._collection!r}: {exc}",
                details={"collection": self._collection},
            ) from exc

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._client.close()

    def _require_collection(self) -> None:
        """Fail with a readable message when the index has not been built yet."""
        if not self._client.collection_exists(self._collection):
            raise IndexNotReadyError(
                f"Qdrant collection {self._collection!r} does not exist. "
                "Run `rag-bench index build` first.",
                details={"collection": self._collection},
            )


def _to_payload(chunk: Chunk) -> dict[str, object]:
    """Flatten a chunk into a Qdrant payload."""
    return {
        "doc_id": chunk.doc_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "section_refs": list(chunk.section_refs),
        "metadata": dict(chunk.metadata),
    }


def _from_payload(point_id: PointId, payload: dict[str, object] | None) -> Chunk:
    """Rebuild a chunk from a Qdrant point.

    The chunk ID is the point ID rather than a payload field, so a chunk survives the
    round trip without storing its identity twice.

    Args:
        point_id: The Qdrant point identifier.
        payload: The stored payload.

    Returns:
        The reconstructed chunk.

    Raises:
        VectorStoreError: If the payload is missing or does not validate, which means the
            collection was written by something other than this store.
    """
    if payload is None:
        raise VectorStoreError(
            f"Qdrant point {point_id} has no payload", details={"point_id": str(point_id)}
        )
    try:
        return Chunk.model_validate({**payload, "id": str(point_id)})
    except ValidationError as exc:
        raise VectorStoreError(
            f"Qdrant point {point_id} is not a valid chunk: {exc}",
            details={"point_id": str(point_id)},
        ) from exc


def _to_scored_chunk(point: ScoredPoint, rank: int) -> ScoredChunk:
    """Turn one Qdrant hit into a scored chunk."""
    return ScoredChunk(
        chunk=_from_payload(point.id, point.payload), score=float(point.score), rank=rank
    )
