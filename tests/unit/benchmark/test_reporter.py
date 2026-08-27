import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import Engine

from rag_bench.benchmark.reporter import (
    latest_run_id,
    load_report,
    to_json,
    to_markdown,
    write_report,
)
from rag_bench.benchmark.runner import BenchmarkRunner
from rag_bench.core.config import SweepConfig
from rag_bench.core.exceptions import ResourceNotFoundError

from .conftest import StubChatModel


@pytest.fixture
def report(sweep: SweepConfig, bench_engine: Engine, chat_model: StubChatModel) -> Iterator[object]:
    with (
        patch("rag_bench.pipeline.querier.check_llm_health"),
        patch("rag_bench.core.llm.build_chat_model", return_value=chat_model),
    ):
        runner = BenchmarkRunner(sweep, engine=bench_engine, pricing=None, check_provider=False)
        run_id = runner.run().run_id
    yield load_report(run_id, bench_engine)


def test_the_report_covers_every_configuration(report) -> None:  # type: ignore[no-untyped-def]
    assert len(report.rows) == 4
    assert {row.chunker for row in report.rows} == {"fixed", "structural"}


def test_ranking_falls_back_when_ragas_was_not_run(report) -> None:  # type: ignore[no-untyped-def]
    # A run without the LLM-judged metrics must still rank by something meaningful.
    assert report.ranking_metric == "hit_rate"


def test_ties_break_on_the_next_metric_not_alphabetically(report) -> None:  # type: ignore[no-untyped-def]
    ranked = report.ranked()
    scores = [(r.value("hit_rate") or 0.0, r.value("mrr") or 0.0) for r in ranked]

    assert scores == sorted(scores, reverse=True)


def test_markdown_emphasises_the_winning_row(report) -> None:  # type: ignore[no-untyped-def]
    rendered = to_markdown(report)
    winner = report.ranked()[0]
    emphasised = [line for line in rendered.splitlines() if line.startswith("| **")]

    # Exactly one row is bolded, and it is the top-ranked one.
    assert len(emphasised) == 1
    assert f"**{winner.chunker}**" in emphasised[0]
    assert f"**{winner.retriever}**" in emphasised[0]


def test_markdown_records_the_commit_and_eval_set(report) -> None:  # type: ignore[no-untyped-def]
    rendered = to_markdown(report)

    assert "- Commit: `" in rendered
    assert "eval.jsonl" in rendered
    assert "3 questions" in rendered


def test_markdown_includes_the_difficulty_breakdown(report) -> None:  # type: ignore[no-untyped-def]
    rendered = to_markdown(report)

    assert "## Hit rate by difficulty" in rendered
    assert "multi_hop" in rendered


def test_markdown_is_a_valid_table(report) -> None:  # type: ignore[no-untyped-def]
    lines = [line for line in to_markdown(report).splitlines() if line.startswith("|")]
    widths = {line.count("|") for line in lines[:6]}

    assert len(widths) == 1


def test_missing_metrics_render_as_not_available(report) -> None:  # type: ignore[no-untyped-def]
    assert "n/a" in to_markdown(report)


def test_json_carries_the_raw_numbers(report) -> None:  # type: ignore[no-untyped-def]
    payload = json.loads(to_json(report))

    assert payload["question_count"] == 3
    assert len(payload["configurations"]) == 4
    assert payload["ranking_metric"] == "hit_rate"
    assert "by_difficulty" in payload["configurations"][0]


def test_writing_produces_both_renderings(report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    markdown_path, json_path = write_report(report, tmp_path / "out")

    assert markdown_path.read_text().startswith("# Benchmark:")
    assert json.loads(json_path.read_text())["name"] == "offline_grid"


def test_the_latest_run_is_findable(report, bench_engine: Engine) -> None:  # type: ignore[no-untyped-def]
    assert latest_run_id(bench_engine) == report.run_id


def test_reporting_an_unknown_run_is_reported(bench_engine: Engine) -> None:
    from uuid import uuid4

    with pytest.raises(ResourceNotFoundError, match="No benchmark run"):
        load_report(uuid4(), bench_engine)


def test_no_runs_at_all_is_reported(bench_engine: Engine) -> None:
    with pytest.raises(ResourceNotFoundError, match="No benchmark runs recorded"):
        latest_run_id(bench_engine)


def test_markdown_names_the_model_that_generated_the_answers(report) -> None:  # type: ignore[no-untyped-def]
    rendered = to_markdown(report)

    assert "- Generated with: `ollama`" in rendered
    assert "llama3.2:3b" in rendered


def test_json_carries_the_generating_model(report) -> None:  # type: ignore[no-untyped-def]
    payload = json.loads(to_json(report))

    assert payload["llm_provider"] == "ollama"
    assert payload["llm_model"] == "llama3.2:3b"
