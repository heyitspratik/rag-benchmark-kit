"""Loader for a directory of Markdown documentation.

This exists to show the pipeline is not built around EU regulations. Point it at any tree
of Markdown files, such as the ``content/en/docs`` directory of the Kubernetes website
repository, and headings become the sections that article numbers are for the primary
corpus. Nothing downstream changes.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_bench.core.exceptions import CorpusError
from rag_bench.core.interfaces import BaseLoader
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Document, DocumentSection
from rag_bench.core.registry import register_loader

logger = get_logger(__name__)

_FRONT_MATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*$", re.MULTILINE)

_MAX_HEADING_LEVEL_AS_SECTION = 3


@register_loader("markdown_docs")
class MarkdownDocsLoader(BaseLoader):
    """Loads every Markdown file under a directory, one document per file."""

    def __init__(self, *, glob: str = "**/*.md", encoding: str = "utf-8") -> None:
        """Initialise the loader.

        Args:
            glob: Pattern used to find files under the corpus root.
            encoding: Text encoding of the source files.
        """
        self._glob = glob
        self._encoding = encoding

    def load(self, root: Path) -> list[Document]:
        """Load every Markdown file under a directory.

        Args:
            root: Directory to search.

        Returns:
            One document per file, in sorted path order, skipping empty files.

        Raises:
            CorpusError: If the directory is missing or contains no matching files.
        """
        if not root.is_dir():
            raise CorpusError(f"No corpus directory at {root}.", details={"path": str(root)})

        paths = sorted(p for p in root.glob(self._glob) if p.is_file())
        if not paths:
            raise CorpusError(
                f"No files matching {self._glob!r} under {root}.",
                details={"path": str(root), "glob": self._glob},
            )

        documents = [
            document for path in paths if (document := self._load_one(root, path)) is not None
        ]
        logger.info("corpus.loaded", corpus="markdown_docs", documents=len(documents))
        return documents

    def _load_one(self, root: Path, path: Path) -> Document | None:
        """Parse one Markdown file, or return None when it holds no prose."""
        raw = path.read_text(encoding=self._encoding, errors="replace")
        text = _FRONT_MATTER_RE.sub("", raw).strip()
        if not text:
            return None

        relative = path.relative_to(root).as_posix()
        return Document(
            id=relative,
            title=_title(text) or relative,
            source=str(path),
            text=text,
            sections=_heading_sections(text, relative),
            metadata={"corpus": "markdown_docs", "path": relative},
        )


def _title(text: str) -> str:
    """The first heading in a document, used as its title."""
    match = _HEADING_RE.search(text)
    return match.group(2).strip() if match else ""


def _heading_sections(text: str, doc_ref: str) -> tuple[DocumentSection, ...]:
    """Turn headings into sections, treating each heading as running to the next one.

    Only the top two heading depths become sections. Deeper headings would produce spans
    too small to retrieve usefully and would blur the level-1 / level-2 split that the
    structural chunker relies on.
    """
    matches = [m for m in _HEADING_RE.finditer(text) if len(m.group(1)) > 1]
    sections = []
    for position, match in enumerate(matches):
        depth = len(match.group(1))
        if depth > _MAX_HEADING_LEVEL_AS_SECTION:
            continue
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        sections.append(
            DocumentSection(
                ref=f"{doc_ref}#{match.group(2).strip()}",
                title=match.group(2).strip(),
                start=match.start(),
                end=end,
                level=depth - 1,
            )
        )
    return tuple(sections)
