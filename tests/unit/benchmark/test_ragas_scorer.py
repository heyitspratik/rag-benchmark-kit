"""RAGAS is optional and needs a live judge, so the integration is covered with fakes.

These tests prove the wiring: what is sent, what comes back, and that a failure costs a
metric rather than a configuration. They cannot prove RAGAS itself grades well.
"""

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from rag_bench.benchmark.metrics import QuestionOutcome
from rag_bench.benchmark.ragas_scorer import RAGAS_METRICS, RagasScorer
from rag_bench.core.exceptions import BenchmarkError

from .test_metrics_helpers import outcome


class _Frame:
    """Stands in for the pandas frame RAGAS returns."""

    def __init__(self, columns: dict[str, list[Any]]) -> None:
        self._columns = columns

    def __getitem__(self, column: str) -> Any:
        return types.SimpleNamespace(iloc=self._columns[column])


@pytest.fixture
def fake_ragas(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a stand-in ragas package for the duration of one test."""
    evaluate = MagicMock()
    ragas = types.ModuleType("ragas")
    ragas.evaluate = evaluate  # type: ignore[attr-defined]
    ragas.EvaluationDataset = lambda samples: samples  # type: ignore[attr-defined]
    ragas.SingleTurnSample = dict  # type: ignore[attr-defined]

    metrics = types.ModuleType("ragas.metrics")
    for name in (
        "Faithfulness",
        "ResponseRelevancy",
        "LLMContextPrecisionWithReference",
        "LLMContextRecall",
    ):
        setattr(metrics, name, MagicMock(return_value=name))

    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics)
    return evaluate


def _scorer(**kwargs: Any) -> RagasScorer:
    return RagasScorer(judge=MagicMock(), embeddings=MagicMock(), **kwargs)


def test_scores_are_attached_to_each_outcome(fake_ragas: MagicMock) -> None:
    fake_ragas.return_value.to_pandas.return_value = _Frame(
        {name: [0.8, 0.4] for name in RAGAS_METRICS}
    )
    outcomes = [outcome(("A",), [("A",)]), outcome(("B",), [("B",)])]

    scored = _scorer().score(outcomes)

    assert scored[0].scores["faithfulness"] == 0.8
    assert scored[1].scores["faithfulness"] == 0.4


def test_existing_scores_are_preserved(fake_ragas: MagicMock) -> None:
    fake_ragas.return_value.to_pandas.return_value = _Frame({name: [0.5] for name in RAGAS_METRICS})
    existing = QuestionOutcome(
        question=outcome(("A",), [("A",)]).question,
        answer=outcome(("A",), [("A",)]).answer,
        scores={"custom": 1.0},
    )

    scored = _scorer().score([existing])

    assert scored[0].scores["custom"] == 1.0
    assert scored[0].scores["faithfulness"] == 0.5


def test_what_is_sent_to_ragas_carries_question_answer_and_context(
    fake_ragas: MagicMock,
) -> None:
    fake_ragas.return_value.to_pandas.return_value = _Frame({name: [0.5] for name in RAGAS_METRICS})

    _scorer().score([outcome(("A",), [("A",)])])

    sample = fake_ragas.call_args.kwargs["dataset"][0]
    assert set(sample) == {"user_input", "response", "retrieved_contexts", "reference"}
    assert sample["retrieved_contexts"]


def test_only_the_requested_metrics_are_computed(fake_ragas: MagicMock) -> None:
    fake_ragas.return_value.to_pandas.return_value = _Frame({"faithfulness": [0.9]})

    scored = _scorer(metrics=["faithfulness"]).score([outcome(("A",), [("A",)])])

    assert set(scored[0].scores) == {"faithfulness"}
    assert fake_ragas.call_args.kwargs["metrics"] == ["Faithfulness"]


def test_a_metric_ragas_could_not_compute_is_skipped(fake_ragas: MagicMock) -> None:
    # RAGAS leaves NaN where it could not grade, and a NaN average poisons the table.
    fake_ragas.return_value.to_pandas.return_value = _Frame(
        {"faithfulness": [float("nan")], "answer_relevancy": [0.7]}
    )

    scored = _scorer(metrics=["faithfulness", "answer_relevancy"]).score(
        [outcome(("A",), [("A",)])]
    )

    assert "faithfulness" not in scored[0].scores
    assert scored[0].scores["answer_relevancy"] == 0.7


def test_scoring_nothing_returns_nothing(fake_ragas: MagicMock) -> None:
    assert _scorer().score([]) == []
    fake_ragas.assert_not_called()


def test_a_ragas_failure_becomes_a_benchmark_error(fake_ragas: MagicMock) -> None:
    fake_ragas.side_effect = RuntimeError("judge unavailable")

    with pytest.raises(BenchmarkError, match="RAGAS evaluation failed"):
        _scorer().score([outcome(("A",), [("A",)])])


def test_an_unknown_metric_is_rejected_at_construction() -> None:
    with pytest.raises(BenchmarkError, match="Unknown RAGAS metric"):
        _scorer(metrics=["faithfulness", "vibes"])


def test_a_missing_ragas_install_names_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ragas", None)

    with pytest.raises(BenchmarkError, match="uv sync --extra ragas"):
        _scorer().score([outcome(("A",), [("A",)])])


def test_the_adapter_presents_a_project_embedder_to_langchain() -> None:
    from rag_bench.benchmark.ragas_scorer import EmbedderAdapter
    from tests.conftest import OfflineEmbedder

    adapter = EmbedderAdapter(OfflineEmbedder())

    assert len(adapter.embed_query("a question")) == 4
    assert len(adapter.embed_documents(["a", "b"])) == 2
