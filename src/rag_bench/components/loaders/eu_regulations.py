"""Loader for EU regulations published in the Official Journal XHTML format.

The Official Journal markup is regular enough to recover the real structure of a
regulation: ``oj-ti-art`` paragraphs carry article numbers, ``oj-sti-art`` their titles,
and numbered paragraphs inside an article begin with ``N.``. Recovering that structure
here, once, means every chunker downstream stays corpus-agnostic, and it is what lets a
chunk produced by any strategy be scored against ground truth written as
``GDPR Art. 15(4)``.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from rag_bench.core.exceptions import CorpusError
from rag_bench.core.interfaces import BaseLoader
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Document, DocumentSection
from rag_bench.core.registry import register_loader

from .download import read_manifest

logger = get_logger(__name__)

BLOCK_SEPARATOR = "\n\n"

_ARTICLE_TITLE_RE = re.compile(r"^Article\s+(\d+[a-z]?)$")
_PARAGRAPH_START_RE = re.compile(r"^(\d{1,3})\.\s")

_ARTICLE_CLASS = "oj-ti-art"
_ARTICLE_SUBTITLE_CLASS = "oj-sti-art"
_BODY_CLASS = "oj-normal"

# An article runs until the next article, chapter heading, or annex title begins.
_DIVISION_CLASSES = frozenset({_ARTICLE_CLASS, "oj-ti-section-1", "oj-doc-ti"})

_ARTICLE_LEVEL = 1
_PARAGRAPH_LEVEL = 2


class _Block:
    """One rendered paragraph, with its offset into the assembled document text."""

    __slots__ = ("classes", "start", "text")

    def __init__(self, start: int, text: str, classes: frozenset[str]) -> None:
        self.start = start
        self.text = text
        self.classes = classes


@register_loader("eu_regulations")
class EuRegulationsLoader(BaseLoader):
    """Parses cached Official Journal XHTML into documents with article-level sections."""

    def __init__(self, *, min_articles: int = 10) -> None:
        """Initialise the loader.

        Args:
            min_articles: Refuse a document yielding fewer articles than this, which
                signals the source markup changed rather than a genuinely short text.
        """
        self._min_articles = min_articles

    def load(self, root: Path) -> list[Document]:
        """Load every regulation named in the corpus manifest.

        Args:
            root: Directory holding the downloaded XHTML and its manifest.

        Returns:
            One document per regulation, with article and paragraph sections.

        Raises:
            CorpusError: If the corpus is absent, or a document cannot be parsed.
        """
        manifest = read_manifest(root)
        entries = manifest.get("documents")
        if not isinstance(entries, list) or not entries:
            raise CorpusError(
                f"Corpus manifest at {root} lists no documents.", details={"path": str(root)}
            )

        documents = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise CorpusError(f"Malformed manifest entry in {root}: {entry!r}")
            documents.append(self._load_one(root, entry))
        return documents

    def _load_one(self, root: Path, entry: dict[str, object]) -> Document:
        """Parse a single regulation described by one manifest entry."""
        doc_id = str(entry.get("doc_id", ""))
        short_name = str(entry.get("short_name", doc_id.upper()))
        path = root / str(entry.get("filename", ""))
        try:
            markup = path.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise CorpusError(
                f"Corpus file listed in the manifest is missing: {path}",
                details={"doc_id": doc_id, "path": str(path)},
            ) from exc

        blocks = _extract_blocks(markup)
        if not blocks:
            raise CorpusError(
                f"No readable text found in {path}. The source markup may have changed.",
                details={"doc_id": doc_id},
            )
        text = BLOCK_SEPARATOR.join(block.text for block in blocks)
        sections = _extract_sections(blocks, len(text), short_name)

        articles = [s for s in sections if s.level == _ARTICLE_LEVEL]
        if len(articles) < self._min_articles:
            raise CorpusError(
                f"Only {len(articles)} articles parsed from {path}, expected at least "
                f"{self._min_articles}. The Official Journal markup may have changed.",
                details={"doc_id": doc_id, "articles": len(articles)},
            )

        logger.info(
            "corpus.loaded",
            doc_id=doc_id,
            chars=len(text),
            articles=len(articles),
            sections=len(sections),
        )
        return Document(
            id=doc_id,
            title=str(entry.get("title", doc_id)),
            source=str(entry.get("url", str(path))),
            text=text,
            sections=sections,
            metadata={"short_name": short_name, "corpus": "eu_regulations"},
        )


def _extract_blocks(markup: str) -> list[_Block]:
    """Flatten the markup into paragraphs, recording where each lands in the joined text.

    Offsets are accumulated while walking rather than searched for afterwards, which keeps
    them exact even where the same sentence appears twice in a document.
    """
    soup = BeautifulSoup(markup, "xml")
    blocks: list[_Block] = []
    offset = 0
    for element in soup.find_all("p"):
        if not isinstance(element, Tag):
            continue
        text = _normalise(element.get_text(" ", strip=True))
        if not text:
            continue
        blocks.append(_Block(offset, text, _classes(element)))
        offset += len(text) + len(BLOCK_SEPARATOR)
    return blocks


def _extract_sections(
    blocks: list[_Block], text_length: int, short_name: str
) -> tuple[DocumentSection, ...]:
    """Derive article and paragraph sections from the classified blocks."""
    division_indexes = [i for i, b in enumerate(blocks) if b.classes & _DIVISION_CLASSES]
    article_indexes = [i for i, b in enumerate(blocks) if _ARTICLE_CLASS in b.classes]

    sections: list[DocumentSection] = []
    for index in article_indexes:
        match = _ARTICLE_TITLE_RE.match(blocks[index].text)
        if match is None:
            continue
        number = match.group(1)
        stop = next((j for j in division_indexes if j > index), len(blocks))
        start = blocks[index].start
        end = _span_end(blocks, stop, text_length)
        sections.append(
            DocumentSection(
                ref=f"{short_name} Art. {number}",
                title=_subtitle(blocks, index),
                start=start,
                end=end,
                level=_ARTICLE_LEVEL,
            )
        )
        sections.extend(
            _paragraph_sections(blocks, index + 1, stop, end, f"{short_name} Art. {number}")
        )
    return tuple(sections)


def _paragraph_sections(
    blocks: list[_Block], start_index: int, stop_index: int, article_end: int, article_ref: str
) -> list[DocumentSection]:
    """Split one article's body into its numbered paragraphs."""
    starts: list[tuple[int, str]] = []
    for i in range(start_index, stop_index):
        match = _PARAGRAPH_START_RE.match(blocks[i].text)
        if match is not None and _BODY_CLASS in blocks[i].classes:
            starts.append((i, match.group(1)))

    sections = []
    for position, (index, number) in enumerate(starts):
        is_last = position + 1 == len(starts)
        end = (
            article_end if is_last else blocks[starts[position + 1][0]].start - len(BLOCK_SEPARATOR)
        )
        sections.append(
            DocumentSection(
                ref=f"{article_ref}({number})",
                start=blocks[index].start,
                end=max(end, blocks[index].start),
                level=_PARAGRAPH_LEVEL,
            )
        )
    return sections


def _span_end(blocks: list[_Block], stop_index: int, text_length: int) -> int:
    """Character offset at which a span ending before ``stop_index`` finishes."""
    if stop_index >= len(blocks):
        return text_length
    return max(blocks[stop_index].start - len(BLOCK_SEPARATOR), 0)


def _subtitle(blocks: list[_Block], article_index: int) -> str:
    """The heading that follows an article number, when the markup carries one."""
    following = article_index + 1
    if following < len(blocks) and _ARTICLE_SUBTITLE_CLASS in blocks[following].classes:
        return blocks[following].text
    return ""


def _classes(element: Tag) -> frozenset[str]:
    """CSS classes of an element, tolerating the XML parser's single-string form."""
    raw = element.get("class")
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        return frozenset(raw.split())
    return frozenset(raw)


def _normalise(text: str) -> str:
    """Collapse Official Journal whitespace, including the non-breaking spaces it uses."""
    return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFKC", text)).strip()
