"""Builders for Official Journal shaped fixtures."""

import json
from pathlib import Path

import pytest

_HEADER = '<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml"><body>'
_FOOTER = "</body></html>"


def build_oj_markup(articles: dict[str, list[str]], recitals: list[str] | None = None) -> str:
    """Build XHTML shaped like an Official Journal regulation.

    An article with a single body paragraph is left unnumbered, as the Official Journal
    does, so fixtures exercise both shapes.

    Args:
        articles: Article number mapped to its paragraph texts.
        recitals: Preamble paragraphs that sit before Article 1.

    Returns:
        The XHTML source.
    """
    parts = [_HEADER]
    for recital in recitals or []:
        parts.append(f'<p class="oj-normal">{recital}</p>')
    for number, paragraphs in articles.items():
        parts.append(f'<p class="oj-ti-art">Article\u00a0{number}</p>')
        parts.append(f'<p class="oj-sti-art">Heading for {number}</p>')
        for position, body in enumerate(paragraphs, start=1):
            # The Official Journal leaves a single-body article unnumbered, so
            # fixtures exercise both the numbered and the unnumbered shape.
            prefix = f"{position}.\u00a0\u00a0\u00a0" if len(paragraphs) > 1 else ""
            parts.append(f'<p class="oj-normal">{prefix}{body}</p>')
    parts.append(_FOOTER)
    return "".join(parts)


def write_corpus(
    root: Path, markup: str, *, doc_id: str = "gdpr", short_name: str = "GDPR"
) -> Path:
    """Write a one-document corpus directory with its manifest.

    Args:
        root: Directory to create the corpus in.
        markup: The document source.
        doc_id: Identifier recorded in the manifest.
        short_name: Citation prefix, such as ``GDPR``.

    Returns:
        The corpus directory.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{doc_id}.xhtml").write_text(markup, encoding="utf-8")
    manifest = {
        "corpus": "eu_regulations",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "documents": [
            {
                "doc_id": doc_id,
                "title": f"Test {short_name}",
                "short_name": short_name,
                "url": f"http://example.invalid/{doc_id}",
                "filename": f"{doc_id}.xhtml",
                "bytes": len(markup),
                "sha256": "0" * 64,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture
def oj_corpus(tmp_path: Path) -> Path:
    """A three-article regulation with a recital, cached like a real download."""
    markup = build_oj_markup(
        recitals=["Whereas the protection of natural persons is a fundamental right."],
        articles={
            "1": ["This Regulation lays down rules.", "It protects fundamental rights."],
            "2": ["This Regulation applies to processing.", "It does not apply to households."],
            "3": ["A single unnumbered body paragraph."],
        },
    )
    return write_corpus(tmp_path / "eu_regulations", markup)
