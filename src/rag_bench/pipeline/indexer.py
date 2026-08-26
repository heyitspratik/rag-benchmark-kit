"""Building an index: load, chunk, annotate, embed, store.

Assembled entirely from names in a config, so this module never mentions a concrete
chunker, embedder, or store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag_bench.components import load_components
from rag_bench.core.config import INDEX_STAGES, PipelineConfig, validate_against_registries
from rag_bench.core.interfaces import BaseChunker, BaseEmbedder, BaseLoader, BaseVectorStore
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Chunk, Document
from rag_bench.core.registry import CHUNKERS, EMBEDDERS, LOADERS, STORES

logger = get_logger(__name__)

#: Chunks embedded and written per round trip, independent of the embedder's own batching.
_UPSERT_BATCH = 256


@dataclass(frozen=True)
class IndexReport:
    """What an index build produced."""

    collection: str
    documents: int
    chunks: int
    dimension: int
    elapsed_s: float
    chunks_per_document: dict[str, int] = field(default_factory=dict)


class Indexer:
    """Turns a corpus into a populated vector collection, following one config."""

    def __init__(self, config: PipelineConfig) -> None:
        """Resolve every component named in the config.

        Args:
            config: A validated pipeline config.

        Raises:
            UnknownComponentError: If the config names something unregistered.
            ConfigValidationError: If a component rejects its parameters.
        """
        load_components()
        validate_against_registries(config, INDEX_STAGES)
        self._config = config
        self._loader: BaseLoader = LOADERS.create(config.corpus.name)
        self._chunker: BaseChunker = CHUNKERS.create(config.chunker.name, config.chunker.params)
        self._embedder: BaseEmbedder = EMBEDDERS.create(
            config.embedder.name, config.embedder.params
        )
        self._store: BaseVectorStore = STORES.create(config.store.name, config.store.params)

    @property
    def store(self) -> BaseVectorStore:
        """The resolved vector store, so callers can close it."""
        return self._store

    def build(self, *, recreate: bool = True) -> IndexReport:
        """Ingest the corpus into the configured collection.

        Args:
            recreate: Drop the collection first. Leave this on unless deliberately adding
                to an existing index; a stale collection silently invalidates a benchmark.

        Returns:
            A summary of what was written.

        Raises:
            CorpusError: If the corpus is missing or unparseable.
            VectorStoreError: If the store rejects a write.
        """
        started = time.perf_counter()
        if recreate:
            self._store.delete_collection()

        documents = self._loader.load(self._config.corpus.path)
        dimension = self._embedder.dimension
        self._store.ensure_collection(dimension)

        per_document: dict[str, int] = {}
        total = 0
        for document in documents:
            chunks = self._chunk(document)
            per_document[document.id] = len(chunks)
            total += len(chunks)
            self._write(chunks)
            logger.info("index.document_indexed", doc_id=document.id, chunks=len(chunks))

        elapsed = time.perf_counter() - started
        report = IndexReport(
            collection=str(self._config.store.params.get("collection", "unknown")),
            documents=len(documents),
            chunks=total,
            dimension=dimension,
            elapsed_s=elapsed,
            chunks_per_document=per_document,
        )
        logger.info(
            "index.built",
            chunker=self._config.chunker.name,
            embedder=self._config.embedder.name,
            documents=report.documents,
            chunks=report.chunks,
            elapsed_s=round(elapsed, 2),
        )
        return report

    def _chunk(self, document: Document) -> list[Chunk]:
        """Chunk a document and annotate each chunk with the sections it covers.

        The annotation lives here rather than in the chunkers so that every strategy,
        including the ones that know nothing about document structure, produces chunks
        that can be scored against section-level ground truth.
        """
        return [
            chunk.with_sections(document.sections_overlapping(chunk.char_start, chunk.char_end))
            for chunk in self._chunker.chunk(document)
        ]

    def _write(self, chunks: list[Chunk]) -> None:
        """Embed and store chunks in batches."""
        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            vectors = self._embedder.embed_documents([chunk.text for chunk in batch])
            self._store.upsert(batch, vectors)
