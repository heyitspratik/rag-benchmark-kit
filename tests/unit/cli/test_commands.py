from pathlib import Path
from unittest.mock import patch

import httpx
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
