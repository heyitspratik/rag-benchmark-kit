"""Local CPU embedders built on sentence-transformers.

All three run offline on CPU with no API key, which is what keeps the quickstart free.
They differ in one respect that matters more than size: the prefixes they expect. BGE
wants an instruction on the query only, E5 wants ``query:`` and ``passage:`` on both
sides. Getting that wrong costs real retrieval quality and is invisible in the output, so
each model owns its own prefixes here rather than leaving them to the call site.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from rag_bench.core.exceptions import ConfigValidationError
from rag_bench.core.interfaces import BaseEmbedder, Vector
from rag_bench.core.logging import get_logger
from rag_bench.core.registry import register_embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

#: BGE models are trained with this instruction on the query side only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder(BaseEmbedder):
    """Shared behaviour for any locally hosted sentence-transformers model."""

    model_id: ClassVar[str] = ""
    query_prefix: ClassVar[str] = ""
    passage_prefix: ClassVar[str] = ""

    def __init__(
        self,
        batch_size: int = 32,
        normalize: bool = True,
        device: str | None = None,
        model_id: str | None = None,
    ) -> None:
        """Initialise the embedder.

        Args:
            batch_size: Passages encoded per forward pass.
            normalize: Return unit vectors, so cosine and dot product agree.
            device: Torch device; sentence-transformers picks one when omitted.
            model_id: Override the class default, for pinning a specific revision.

        Raises:
            ConfigValidationError: If the batch size is not positive.
        """
        if batch_size <= 0:
            raise ConfigValidationError(
                f"embedder.params.batch_size must be positive, got {batch_size}"
            )
        self._batch_size = batch_size
        self._normalize = normalize
        self._device = device
        self._model_id = model_id or self.model_id
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        """Vector length reported by the loaded model."""
        size = self._load().get_embedding_dimension()
        if size is None:
            raise ConfigValidationError(
                f"Model {self._model_id!r} does not report an embedding dimension"
            )
        return int(size)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed passages, applying this model's passage prefix.

        Args:
            texts: Passage texts.

        Returns:
            One vector per input, in the same order.
        """
        if not texts:
            return []
        return self._encode([f"{self.passage_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> Vector:
        """Embed a query, applying this model's query prefix.

        Args:
            text: The user's question.

        Returns:
            The query vector.
        """
        return self._encode([f"{self.query_prefix}{text}"])[0]

    def _encode(self, texts: list[str]) -> list[Vector]:
        """Run the model over already-prefixed text."""
        encoded = self._load().encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in encoded]

    def _load(self) -> SentenceTransformer:
        """Load the model on first use.

        Imported lazily because sentence-transformers pulls in torch, which costs seconds
        of import time that the CLI should not pay when it is only printing help.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("embedder.loading", model=self._model_id, device=self._device)
            self._model = SentenceTransformer(self._model_id, device=self._device)
        return self._model


@register_embedder("bge_small")
class BgeSmallEmbedder(SentenceTransformerEmbedder):
    """BAAI/bge-small-en-v1.5. The default: fast on CPU and good enough to be a baseline."""

    model_id: ClassVar[str] = "BAAI/bge-small-en-v1.5"
    query_prefix: ClassVar[str] = BGE_QUERY_INSTRUCTION


@register_embedder("bge_large")
class BgeLargeEmbedder(SentenceTransformerEmbedder):
    """BAAI/bge-large-en-v1.5. Slower and larger, to show what the extra size buys."""

    model_id: ClassVar[str] = "BAAI/bge-large-en-v1.5"
    query_prefix: ClassVar[str] = BGE_QUERY_INSTRUCTION


@register_embedder("e5_base")
class E5BaseEmbedder(SentenceTransformerEmbedder):
    """intfloat/e5-base-v2, which is trained to expect asymmetric prefixes on both sides."""

    model_id: ClassVar[str] = "intfloat/e5-base-v2"
    query_prefix: ClassVar[str] = "query: "
    passage_prefix: ClassVar[str] = "passage: "
