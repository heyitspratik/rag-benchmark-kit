"""Optional RAGAS scoring.

These four metrics are judged by a language model rather than computed, which is why
they are optional and why the harness never depends on them. They carry the judge's own
variance, they cost a model call per question per configuration, and a run without them
still produces hit rate, MRR, abstention accuracy, latency and cost.

Kept in its own module so that ``import rag_bench.benchmark`` never pulls in the RAGAS
dependency tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings

from rag_bench.benchmark.metrics import QuestionOutcome
from rag_bench.core.exceptions import BenchmarkError
from rag_bench.core.interfaces import BaseEmbedder
from rag_bench.core.logging import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)

#: The metrics this scorer produces, matching the columns on ``configuration_metrics``.
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

_INSTALL_HINT = "Install it with `uv sync --extra ragas`."


class EmbedderAdapter(Embeddings):
    """Presents a project embedder as the LangChain type RAGAS expects.

    RAGAS is built around LangChain's interfaces, and this repo is not. Adapting at the
    boundary keeps that dependency out of the embedder implementations themselves.
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        """Wrap an embedder.

        Args:
            embedder: The embedder to expose.
        """
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages.

        Args:
            texts: Passage texts.

        Returns:
            One vector per input.
        """
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        Args:
            text: The query.

        Returns:
            The query vector.
        """
        return self._embedder.embed_query(text)


class RagasScorer:
    """Scores answered questions with RAGAS, leaving the outcomes otherwise untouched."""

    def __init__(
        self,
        judge: BaseChatModel,
        embeddings: Embeddings,
        metrics: Sequence[str] = RAGAS_METRICS,
    ) -> None:
        """Initialise the scorer.

        Args:
            judge: The chat model RAGAS uses to grade answers. Use a stronger model than
                the one under test, or the grades measure the same weaknesses twice.
            embeddings: Embedding model RAGAS uses for its similarity-based metrics.
            metrics: Which of :data:`RAGAS_METRICS` to compute.

        Raises:
            BenchmarkError: If an unknown metric is requested.
        """
        unknown = sorted(set(metrics) - set(RAGAS_METRICS))
        if unknown:
            raise BenchmarkError(
                f"Unknown RAGAS metric(s): {', '.join(unknown)}. "
                f"Available: {', '.join(RAGAS_METRICS)}",
                details={"requested": list(metrics)},
            )
        self._judge = judge
        self._embeddings = embeddings
        self._metrics = tuple(metrics)

    def score(self, outcomes: Sequence[QuestionOutcome]) -> list[QuestionOutcome]:
        """Grade every outcome, returning copies with their scores filled in.

        Args:
            outcomes: Answered questions for one configuration.

        Returns:
            The same outcomes with per-question RAGAS scores attached. A question RAGAS
            cannot grade keeps its existing scores rather than being dropped, so a
            partial failure costs a metric and not a configuration.

        Raises:
            BenchmarkError: If RAGAS is not installed, or the evaluation itself failed.
        """
        if not outcomes:
            return []

        try:
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
        except ImportError as exc:
            raise BenchmarkError(f"RAGAS is not installed. {_INSTALL_HINT}") from exc

        samples = [
            SingleTurnSample(
                user_input=outcome.question.question,
                response=outcome.answer.text,
                retrieved_contexts=[c.chunk.text for c in outcome.answer.contexts],
                reference=outcome.question.ground_truth,
            )
            for outcome in outcomes
        ]

        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=samples),
                metrics=self._metric_objects(),
                llm=self._judge,
                embeddings=self._embeddings,
            )
            frame = result.to_pandas()
        except (ImportError, ValueError, RuntimeError, KeyError, AttributeError) as exc:
            raise BenchmarkError(f"RAGAS evaluation failed: {exc}") from exc

        logger.info("ragas.scored", questions=len(outcomes), metrics=list(self._metrics))
        return [
            replace(outcome, scores={**outcome.scores, **self._row(frame, position)})
            for position, outcome in enumerate(outcomes)
        ]

    def _metric_objects(self) -> list[Any]:
        """Build the RAGAS metric instances this scorer was configured with.

        Raises:
            BenchmarkError: If RAGAS is not installed.
        """
        try:
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                ResponseRelevancy,
            )
        except ImportError as exc:
            raise BenchmarkError(f"RAGAS is not installed. {_INSTALL_HINT}") from exc

        available: dict[str, Any] = {
            "faithfulness": Faithfulness,
            "answer_relevancy": ResponseRelevancy,
            "context_precision": LLMContextPrecisionWithReference,
            "context_recall": LLMContextRecall,
        }
        return [available[name]() for name in self._metrics]

    def _row(self, frame: Any, position: int) -> dict[str, float]:
        """Pull one question's scores out of the RAGAS result frame, skipping blanks."""
        scores: dict[str, float] = {}
        for metric in self._metrics:
            value = _cell(frame, metric, position)
            if value is not None:
                scores[metric] = value
        return scores


def _cell(frame: Any, column: str, position: int) -> float | None:
    """Read one cell from a RAGAS result frame, tolerating a missing or unusable value."""
    try:
        value = frame[column].iloc[position]
    except (KeyError, IndexError, AttributeError, TypeError):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # RAGAS leaves NaN where a metric could not be computed for a question.
    return None if number != number else number
