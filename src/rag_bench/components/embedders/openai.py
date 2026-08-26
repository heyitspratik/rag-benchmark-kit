"""OpenAI embeddings, the one paid option.

Included so the benchmark can answer whether a hosted model is worth its cost, but never
the default: the quickstart and the test suite must run without a key.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rag_bench.core.exceptions import ConfigValidationError, LLMProviderError
from rag_bench.core.interfaces import BaseEmbedder, Vector
from rag_bench.core.registry import register_embedder
from rag_bench.core.settings import get_settings

if TYPE_CHECKING:
    from openai import OpenAI

#: Vector length of the default model, used before the first call has been made.
_KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@register_embedder("openai")
class OpenAIEmbedder(BaseEmbedder):
    """Embeds through the OpenAI API.

    The API is symmetric, so unlike the local models there are no query or passage
    prefixes to apply.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 128,
        dimensions: int | None = None,
    ) -> None:
        """Initialise the embedder.

        Args:
            model: OpenAI embedding model name.
            batch_size: Inputs per request.
            dimensions: Truncate to this many dimensions, where the model supports it.

        Raises:
            ConfigValidationError: If the batch size is not positive, or the vector length
                for an unfamiliar model cannot be determined.
        """
        if batch_size <= 0:
            raise ConfigValidationError(
                f"embedder.params.batch_size must be positive, got {batch_size}"
            )
        if dimensions is None and model not in _KNOWN_DIMENSIONS:
            raise ConfigValidationError(
                f"Unknown vector length for OpenAI model {model!r}. "
                "Set embedder.params.dimensions explicitly."
            )
        self._model = model
        self._batch_size = batch_size
        self._dimensions = dimensions or _KNOWN_DIMENSIONS[model]
        self._client: OpenAI | None = None

    @property
    def dimension(self) -> int:
        """Vector length this embedder produces."""
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch of passages.

        Args:
            texts: Passage texts.

        Returns:
            One vector per input, in the same order.
        """
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed(list(texts[start : start + self._batch_size])))
        return vectors

    def embed_query(self, text: str) -> Vector:
        """Embed a single query.

        Args:
            text: The user's question.

        Returns:
            The query vector.
        """
        return self._embed([text])[0]

    def _embed(self, batch: list[str]) -> list[Vector]:
        """Send one request, preserving input order."""
        response = self._connect().embeddings.create(
            model=self._model, input=batch, dimensions=self._dimensions
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [[float(value) for value in item.embedding] for item in ordered]

    def _connect(self) -> OpenAI:
        """Build the API client on first use.

        Raises:
            LLMProviderError: If no OpenAI key is configured.
        """
        if self._client is None:
            from openai import OpenAI

            key = get_settings().llm.openai_api_key
            if key is None:
                raise LLMProviderError(
                    "OPENAI_API_KEY is required to use the 'openai' embedder",
                    details={"embedder": "openai"},
                )
            self._client = OpenAI(api_key=key.get_secret_value())
        return self._client
