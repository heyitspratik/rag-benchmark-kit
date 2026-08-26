"""Fetching corpora onto disk, kept separate from parsing them.

The EU texts come from the Publications Office Cellar resource API rather than the
EUR-Lex web pages. The web pages sit behind a JavaScript bot challenge that no plain HTTP
client can pass, whereas Cellar serves the same authenticated XHTML through ordinary
content negotiation. Downloads are cached and content-addressed in a manifest so a corpus
can be traced back to exactly what was fetched and when.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from rag_bench.core.exceptions import CorpusError
from rag_bench.core.logging import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"

_CELLAR_URL = "http://publications.europa.eu/resource/celex/{celex}"
_CELLAR_HEADERS = {"Accept": "application/xhtml+xml", "Accept-Language": "eng"}
_DEFAULT_TIMEOUT_S = 180.0

# Below this the response is a Cellar error page rather than a regulation.
_MIN_PLAUSIBLE_BYTES = 50_000


@dataclass(frozen=True)
class RemoteDocument:
    """One downloadable source document."""

    doc_id: str
    title: str
    short_name: str
    url: str
    filename: str


EU_REGULATIONS = (
    RemoteDocument(
        doc_id="gdpr",
        title="Regulation (EU) 2016/679 (General Data Protection Regulation)",
        short_name="GDPR",
        url=_CELLAR_URL.format(celex="32016R0679"),
        filename="gdpr.xhtml",
    ),
    RemoteDocument(
        doc_id="ai_act",
        title="Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
        short_name="AI Act",
        url=_CELLAR_URL.format(celex="32024R1689"),
        filename="ai_act.xhtml",
    ),
)

CORPUS_SOURCES: dict[str, tuple[RemoteDocument, ...]] = {"eu_regulations": EU_REGULATIONS}


def download_corpus(
    name: str,
    destination: Path,
    *,
    force: bool = False,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> list[Path]:
    """Download a corpus into a cache directory, skipping files already present.

    Args:
        name: Corpus name, a key of :data:`CORPUS_SOURCES`.
        destination: Directory to cache into; created if absent.
        force: Re-download even when the file is already cached.
        timeout_s: Per-request timeout.

    Returns:
        The cached file paths, in source order.

    Raises:
        CorpusError: If the corpus is unknown or a document could not be fetched.
    """
    sources = CORPUS_SOURCES.get(name)
    if sources is None:
        raise CorpusError(
            f"Unknown corpus {name!r}. Available: {', '.join(sorted(CORPUS_SOURCES))}",
            details={"requested": name, "available": sorted(CORPUS_SOURCES)},
        )

    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str | int]] = []
    paths: list[Path] = []

    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        for source in sources:
            path = destination / source.filename
            if path.exists() and not force:
                logger.info("corpus.cached", doc_id=source.doc_id, path=str(path))
            else:
                path.write_bytes(_fetch(client, source))
                logger.info("corpus.downloaded", doc_id=source.doc_id, bytes=path.stat().st_size)
            payload = path.read_bytes()
            entries.append(
                {
                    "doc_id": source.doc_id,
                    "title": source.title,
                    "short_name": source.short_name,
                    "url": source.url,
                    "filename": source.filename,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            paths.append(path)

    manifest = {
        "corpus": name,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "documents": entries,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def read_manifest(directory: Path) -> dict[str, object]:
    """Read the manifest written alongside a downloaded corpus.

    Args:
        directory: The corpus cache directory.

    Returns:
        The parsed manifest.

    Raises:
        CorpusError: If the manifest is missing or unreadable.
    """
    path = directory / MANIFEST_NAME
    try:
        parsed: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusError(
            f"No corpus found at {directory}. Run `rag-bench corpus download` first.",
            details={"path": str(directory)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise CorpusError(f"Corpus manifest at {path} is not valid JSON: {exc}") from exc
    return parsed


def _fetch(client: httpx.Client, source: RemoteDocument) -> bytes:
    """Fetch one document, failing loudly on a response too small to be the real text."""
    try:
        response = client.get(source.url, headers=_CELLAR_HEADERS)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise CorpusError(
            f"Could not download {source.doc_id} from {source.url}: {exc}",
            details={"doc_id": source.doc_id, "url": source.url},
        ) from exc

    if len(response.content) < _MIN_PLAUSIBLE_BYTES:
        raise CorpusError(
            f"{source.doc_id} came back as {len(response.content)} bytes, which is too small "
            f"to be the regulation. The source may have changed or be rate limiting.",
            details={"doc_id": source.doc_id, "bytes": len(response.content)},
        )
    return response.content
