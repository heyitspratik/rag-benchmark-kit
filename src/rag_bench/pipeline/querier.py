"""Answering a question: retrieve, then generate, timing each separately.

Retrieval and generation are timed apart rather than together because they are the two
knobs a reader is choosing between. A configuration that answers slightly better but
spends three times as long reranking is a different trade from one that spends it in the
model, and an aggregate latency figure hides which.
"""

from __future__ import annotations

import time

from rag_bench.components import load_components
from rag_bench.core.config import ALL_STAGES, PipelineConfig, validate_against_registries
from rag_bench.core.interfaces import (
    BaseEmbedder,
    BaseGenerator,
    BaseRetriever,
    BaseVectorStore,
)
from rag_bench.core.llm import check_llm_health
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Answer
from rag_bench.core.registry import EMBEDDERS, GENERATORS, RETRIEVERS, STORES
from rag_bench.core.settings import get_settings

logger = get_logger(__name__)

_MS_PER_SECOND = 1000.0


class Querier:
    """Answers questions against an already-built index, following one config."""

    def __init__(self, config: PipelineConfig, *, check_provider: bool = True) -> None:
        """Resolve every component the query path needs.

        Args:
            config: A validated pipeline config.
            check_provider: Probe the LLM provider up front. Leave this on: a benchmark
                that discovers Ollama is down on question 40 of 100 has wasted the run.

        Raises:
            UnknownComponentError: If the config names something unregistered.
            LLMProviderError: If the provider is unreachable or its model is not pulled.
        """
        load_components()
        validate_against_registries(config, ALL_STAGES)
        self._config = config

        self._embedder: BaseEmbedder = EMBEDDERS.create(
            config.embedder.name, config.embedder.params
        )
        self._store: BaseVectorStore = STORES.create(config.store.name, config.store.params)
        # The retriever's collaborators are injected through the same params mapping the
        # registry already uses, so dependency wiring needs no second mechanism.
        self._retriever: BaseRetriever = RETRIEVERS.create(
            config.retriever.name,
            {**config.retriever.params, "embedder": self._embedder, "store": self._store},
        )
        self._generator: BaseGenerator = GENERATORS.create(
            config.generator.name, config.generator.params
        )

        if check_provider:
            check_llm_health(get_settings().llm)

    @property
    def store(self) -> BaseVectorStore:
        """The resolved vector store, so callers can close it."""
        return self._store

    @property
    def retriever(self) -> BaseRetriever:
        """The resolved retriever, so the benchmark can reuse it across questions."""
        return self._retriever

    def answer(self, question: str, k: int | None = None) -> Answer:
        """Retrieve context for a question and generate a cited answer.

        Args:
            question: The user's question.
            k: How many chunks to retrieve; falls back to the retriever's ``top_k``.

        Returns:
            The answer, with per-stage latencies recorded.

        Raises:
            IndexNotReadyError: If no index has been built for this configuration.
            CitationError: If the answer cites context that was not retrieved.
            LLMProviderError: If the provider call failed.
        """
        started = time.perf_counter()
        contexts = self._retriever.retrieve(question, k)
        retrieved_at = time.perf_counter()

        answer = self._generator.generate(question, contexts)
        finished = time.perf_counter()

        timed = answer.model_copy(
            update={
                "retrieval_ms": (retrieved_at - started) * _MS_PER_SECOND,
                "generation_ms": (finished - retrieved_at) * _MS_PER_SECOND,
            }
        )
        logger.info(
            "query.answered",
            retriever=self._config.retriever.name,
            contexts=len(contexts),
            citations=len(timed.citations),
            abstained=timed.abstained,
            retrieval_ms=round(timed.retrieval_ms, 1),
            generation_ms=round(timed.generation_ms, 1),
        )
        return timed
