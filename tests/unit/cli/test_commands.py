from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from rag_bench.cli.main import app
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
