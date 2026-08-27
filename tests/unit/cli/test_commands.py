from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from rag_bench.cli.main import app
from rag_bench.core.settings import get_settings
from rag_bench.db.session import get_engine
from tests.conftest import OFFLINE_EMBEDDER

runner = CliRunner()


def _response(size: int = 100_000) -> httpx.Response:
    request = httpx.Request("GET", "http://example.invalid/doc")
    return httpx.Response(200, content=b"x" * size, request=request)


def test_corpus_list_names_the_documents() -> None:
    result = runner.invoke(app, ["corpus", "list"])

    assert result.exit_code == 0
    assert "eu_regulations" in result.output
    assert "gdpr" in result.output


def test_corpus_download_reports_what_it_cached(tmp_path: Path) -> None:
    with patch("httpx.Client.get", return_value=_response()):
        result = runner.invoke(app, ["corpus", "download", "--dest", str(tmp_path)])

    assert result.exit_code == 0
    assert "documents" in result.output
    assert (tmp_path / "manifest.json").exists()


def test_a_project_error_prints_its_code_and_exits_non_zero(tmp_path: Path) -> None:
    # A missing corpus is a user mistake, so it deserves a message rather than a
    # stack trace.
    result = runner.invoke(app, ["corpus", "download", "--corpus", "nope", "--dest", str(tmp_path)])

    assert result.exit_code == 1
    assert "CORPUS_ERROR" in result.output


def test_index_build_reports_an_invalid_config(tmp_path: Path) -> None:
    config = tmp_path / "broken.yaml"
    config.write_text("corpus:\n  name: x\n  path: y\n")

    result = runner.invoke(app, ["index", "build", "--config", str(config)])

    assert result.exit_code == 1
    assert "CONFIG_INVALID" in result.output


def test_index_build_reports_a_missing_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", "build", "--config", str(tmp_path / "absent.yaml")])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_subcommands_are_discoverable_from_the_root_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert "corpus" in result.output
    assert "index" in result.output


def _offline_config(tmp_path: Path) -> Path:
    """Write a pipeline config wired entirely to offline components."""
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "page.md").write_text("# Title\n\nSome prose.\n\n## Fees\n\nA reasonable fee.\n")
    config = tmp_path / "offline.yaml"
    config.write_text(
        f"corpus:\n"
        f"  name: markdown_docs\n"
        f"  path: {corpus}\n"
        f"chunker:\n"
        f"  name: structural\n"
        f"  params: {{max_chars: 200, overlap: 0}}\n"
        f"embedder:\n"
        f"  name: {OFFLINE_EMBEDDER}\n"
        f"store:\n"
        f"  name: qdrant\n"
        f"  params: {{collection: cli_test, url: 'file:{tmp_path / 'qdrant'}'}}\n"
        f"retriever:\n"
        f"  name: dense\n"
        f"generator:\n"
        f"  name: cited\n"
    )
    return config


def test_index_build_reports_what_it_wrote(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", "build", "--config", str(_offline_config(tmp_path))])

    assert result.exit_code == 0, result.output
    assert "cli_test" in result.output
    assert "structural" in result.output


def test_index_status_reports_the_stored_chunk_count(tmp_path: Path) -> None:
    config = _offline_config(tmp_path)
    runner.invoke(app, ["index", "build", "--config", str(config)])

    result = runner.invoke(app, ["index", "status", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "chunks" in result.output


class _StubChatModel:
    """A provider stand-in, so the CLI tests never need a key or a running Ollama."""

    def invoke(self, messages: list[tuple[str, str]]) -> AIMessage:
        return AIMessage(content="A controller may charge a reasonable fee [1].")


def _built_config(tmp_path: Path) -> Path:
    """Write an offline config and build its index, ready to query."""
    config = _offline_config(tmp_path)
    runner.invoke(app, ["index", "build", "--config", str(config)])
    return config


def _stubbed_provider() -> Any:
    return patch.multiple(
        "rag_bench.pipeline.querier",
        check_llm_health=lambda *_args, **_kwargs: None,
    )


def test_query_prints_the_answer_and_its_sources(tmp_path: Path) -> None:
    config = _built_config(tmp_path)

    with (
        _stubbed_provider(),
        patch("rag_bench.core.llm.build_chat_model", return_value=_StubChatModel()),
    ):
        result = runner.invoke(app, ["query", "What fee may be charged?", "-c", str(config)])

    assert result.exit_code == 0, result.output
    assert "reasonable fee" in result.output
    assert "Sources" in result.output


def test_query_can_show_the_retrieved_context(tmp_path: Path) -> None:
    config = _built_config(tmp_path)

    with (
        _stubbed_provider(),
        patch("rag_bench.core.llm.build_chat_model", return_value=_StubChatModel()),
    ):
        result = runner.invoke(
            app, ["query", "What fee?", "-c", str(config), "--show-context", "--k", "1"]
        )

    assert result.exit_code == 0, result.output
    assert "Retrieved context" in result.output
    assert "score=" in result.output


def test_query_reports_an_unreachable_provider(tmp_path: Path) -> None:
    # The whole point of the startup probe: a clean message, not a stack trace from
    # somewhere deep inside a retrieval call.
    config = _built_config(tmp_path)

    result = runner.invoke(app, ["query", "anything", "-c", str(config)])

    assert result.exit_code == 1
    assert "LLM_PROVIDER_ERROR" in result.output


def test_query_reports_a_missing_index(tmp_path: Path) -> None:
    config = _offline_config(tmp_path)

    with _stubbed_provider():
        result = runner.invoke(app, ["query", "anything", "-c", str(config)])

    assert result.exit_code == 1
    assert "INDEX_NOT_READY" in result.output


def test_benchmark_plan_reports_the_index_saving() -> None:
    # The plan is what shows a 24-run grid needs only 8 ingestions.
    result = runner.invoke(
        app,
        [
            "benchmark",
            "plan",
            "-e",
            "configs/experiments/full_grid.yaml",
            "--eval-set",
            "data/eval/smoke.jsonl",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "24" in result.output
    assert "ingestions saved" in result.output


def test_benchmark_plan_reports_an_invalid_sweep(tmp_path: Path) -> None:
    sweep = tmp_path / "bad.yaml"
    sweep.write_text("name: x\neval_set: e.jsonl\nbase_config: b.yaml\nsweep:\n  chunker: []\n")

    result = runner.invoke(app, ["benchmark", "plan", "-e", str(sweep)])

    assert result.exit_code == 1
    assert "CONFIG_INVALID" in result.output


def test_benchmark_report_without_a_database_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_DSN", "postgresql+psycopg://x@127.0.0.1:1/none")
    get_settings.cache_clear()
    get_engine.cache_clear()

    result = runner.invoke(app, ["benchmark", "report"])

    assert result.exit_code == 1
    assert "DATABASE_ERROR" in result.output


def _benchmark_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the CLI at a throwaway database with the schema in place."""
    from sqlalchemy import create_engine

    from rag_bench.db.models import Base

    dsn = f"sqlite:///{tmp_path / 'bench.db'}"
    engine = create_engine(dsn)
    Base.metadata.create_all(engine)
    engine.dispose()

    monkeypatch.setenv("POSTGRES_DSN", dsn)
    get_settings.cache_clear()
    get_engine.cache_clear()


def test_benchmark_report_with_no_runs_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _benchmark_database(tmp_path, monkeypatch)

    result = runner.invoke(app, ["benchmark", "report"])

    assert result.exit_code == 1
    assert "NOT_FOUND" in result.output


def test_benchmark_report_renders_a_stored_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _benchmark_database(tmp_path, monkeypatch)
    _seed_run(tmp_path)

    result = runner.invoke(app, ["benchmark", "report"])

    assert result.exit_code == 0, result.output
    assert "# Benchmark: seeded" in result.output
    assert "| Chunker |" in result.output


def test_benchmark_report_can_emit_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _benchmark_database(tmp_path, monkeypatch)
    _seed_run(tmp_path)

    result = runner.invoke(app, ["benchmark", "report", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert "ranking_metric" in result.output


def test_benchmark_report_can_write_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _benchmark_database(tmp_path, monkeypatch)
    _seed_run(tmp_path)
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    get_settings.cache_clear()

    result = runner.invoke(app, ["benchmark", "report", "--write"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "seeded" / "results.md").exists()
    assert (tmp_path / "results" / "seeded" / "results.json").exists()


def _seed_run(tmp_path: Path) -> None:
    """Insert one finished configuration, so report has something to render."""
    from rag_bench.db.models import (
        BenchmarkRun,
        ConfigurationMetrics,
        RunConfiguration,
        RunStatus,
    )
    from rag_bench.db.session import get_engine as engine_factory
    from rag_bench.db.session import session_scope

    with session_scope(engine_factory()) as session:
        run = BenchmarkRun(
            name="seeded",
            git_sha="c" * 40,
            status=RunStatus.COMPLETED,
            eval_set="data/eval/smoke.jsonl",
            question_count=10,
            sweep_config={"chunker": ["structural"]},
        )
        session.add(run)
        session.flush()
        configuration = RunConfiguration(
            run_id=run.id,
            fingerprint="f" * 16,
            index_fingerprint="i" * 16,
            corpus="eu_regulations",
            chunker="structural",
            embedder="bge_small",
            store="qdrant",
            retriever="hybrid",
            generator="cited",
            resolved_config={},
            status=RunStatus.COMPLETED,
        )
        session.add(configuration)
        session.flush()
        session.add(
            ConfigurationMetrics(
                configuration_id=configuration.id,
                hit_rate=0.9,
                mrr=0.75,
                question_count=10,
                by_difficulty={"multi_hop": {"hit_rate": 0.6}},
            )
        )


def test_eval_stats_describes_the_committed_set() -> None:
    result = runner.invoke(app, ["eval", "stats"])

    assert result.exit_code == 0, result.output
    assert "questions" in result.output
    assert "negative" in result.output


def test_eval_verify_reports_a_broken_citation(tmp_path: Path) -> None:
    # A ref naming nothing would make hit rate unreachable for that question forever,
    # so it must fail loudly rather than be reported as a passing set.
    corpus = _markdown_corpus(tmp_path)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"id": "q1", "question": "What fee?", "ground_truth": "A reasonable fee.", '
        '"source_refs": ["nowhere.md#Nothing"], "difficulty": "single_hop"}\n'
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "verify",
            "-e",
            str(bad),
            "--corpus",
            "markdown_docs",
            "--corpus-path",
            str(corpus),
        ],
    )

    assert result.exit_code == 1
    assert "not a section of the corpus" in result.output


def test_eval_verify_passes_a_grounded_set(tmp_path: Path) -> None:
    corpus = _markdown_corpus(tmp_path)
    good = tmp_path / "good.jsonl"
    good.write_text(
        '{"id": "q1", "question": "What fee may a controller charge?", '
        '"ground_truth": "A controller may charge a reasonable fee based on '
        'administrative costs.", "source_refs": ["controllers.md#Fees"], '
        '"difficulty": "single_hop"}\n'
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "verify",
            "-e",
            str(good),
            "--corpus",
            "markdown_docs",
            "--corpus-path",
            str(corpus),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Every citation resolves" in result.output


def _markdown_corpus(tmp_path: Path) -> Path:
    """A tiny corpus whose headings become citable sections."""
    corpus = tmp_path / "docs"
    corpus.mkdir(exist_ok=True)
    (corpus / "controllers.md").write_text(
        "# Controllers\n\nProse about controllers.\n\n## Fees\n\n"
        "A controller may charge a reasonable fee based on administrative costs.\n"
    )
    return corpus
