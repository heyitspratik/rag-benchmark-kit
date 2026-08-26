import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from rag_bench.components.loaders.download import (
    MANIFEST_NAME,
    download_corpus,
    read_manifest,
)
from rag_bench.core.exceptions import CorpusError

_BODY = b"x" * 100_000


def _response(content: bytes = _BODY, status: int = 200) -> httpx.Response:
    request = httpx.Request("GET", "http://example.invalid/doc")
    return httpx.Response(status, content=content, request=request)


def test_download_writes_files_and_a_manifest(tmp_path: Path) -> None:
    with patch("httpx.Client.get", return_value=_response()):
        paths = download_corpus("eu_regulations", tmp_path)

    assert [p.name for p in paths] == ["gdpr.xhtml", "ai_act.xhtml"]
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["corpus"] == "eu_regulations"
    assert {d["doc_id"] for d in manifest["documents"]} == {"gdpr", "ai_act"}


def test_manifest_records_a_checksum_for_traceability(tmp_path: Path) -> None:
    with patch("httpx.Client.get", return_value=_response()):
        download_corpus("eu_regulations", tmp_path)

    entry = json.loads((tmp_path / MANIFEST_NAME).read_text())["documents"][0]
    assert len(entry["sha256"]) == 64
    assert entry["bytes"] == len(_BODY)


def test_cached_files_are_not_refetched(tmp_path: Path) -> None:
    with patch("httpx.Client.get", return_value=_response()) as get:
        download_corpus("eu_regulations", tmp_path)
        download_corpus("eu_regulations", tmp_path)

    assert get.call_count == 2


def test_force_refetches(tmp_path: Path) -> None:
    with patch("httpx.Client.get", return_value=_response()) as get:
        download_corpus("eu_regulations", tmp_path)
        download_corpus("eu_regulations", tmp_path, force=True)

    assert get.call_count == 4


def test_unknown_corpus_lists_what_exists(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="Unknown corpus 'nope'"):
        download_corpus("nope", tmp_path)


def test_a_transport_failure_becomes_a_corpus_error(tmp_path: Path) -> None:
    with (
        patch("httpx.Client.get", side_effect=httpx.ConnectError("refused")),
        pytest.raises(CorpusError, match="Could not download"),
    ):
        download_corpus("eu_regulations", tmp_path)


def test_a_suspiciously_small_response_is_rejected(tmp_path: Path) -> None:
    # A bot challenge page returns 200 with a couple of kilobytes of JavaScript, which
    # would otherwise be cached and parsed as if it were the regulation.
    with (
        patch("httpx.Client.get", return_value=_response(b"<html>challenge</html>")),
        pytest.raises(CorpusError, match="too small"),
    ):
        download_corpus("eu_regulations", tmp_path)


def test_read_manifest_points_at_the_download_command(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="rag-bench corpus download"):
        read_manifest(tmp_path)


def test_read_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).write_text("{not json")

    with pytest.raises(CorpusError, match="not valid JSON"):
        read_manifest(tmp_path)
