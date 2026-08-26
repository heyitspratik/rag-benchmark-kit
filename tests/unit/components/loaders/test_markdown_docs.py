from pathlib import Path

import pytest

from rag_bench.components.loaders.markdown_docs import MarkdownDocsLoader
from rag_bench.core.exceptions import CorpusError

_PAGE = """---
title: Pods
weight: 10
---

# Pods

Intro text about pods.

## Pod lifecycle

A Pod moves through phases.

### Restart policy

Sub-headings nest one level down.

#### Backoff detail

Headings below the third level are too small to retrieve.

## Pod networking

Each Pod gets an IP address.
"""


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "concepts").mkdir(parents=True)
    (root / "concepts" / "pods.md").write_text(_PAGE)
    (root / "empty.md").write_text("---\ntitle: x\n---\n")
    return root


def test_one_document_per_file(tmp_path: Path) -> None:
    documents = MarkdownDocsLoader().load(_corpus(tmp_path))

    assert [d.id for d in documents] == ["concepts/pods.md"]


def test_front_matter_is_stripped(tmp_path: Path) -> None:
    document = MarkdownDocsLoader().load(_corpus(tmp_path))[0]

    assert "weight: 10" not in document.text
    assert document.text.startswith("# Pods")


def test_the_first_heading_becomes_the_title(tmp_path: Path) -> None:
    document = MarkdownDocsLoader().load(_corpus(tmp_path))[0]

    assert document.title == "Pods"


def test_headings_become_sections_with_truthful_offsets(tmp_path: Path) -> None:
    document = MarkdownDocsLoader().load(_corpus(tmp_path))[0]

    lifecycle = next(s for s in document.sections if s.title == "Pod lifecycle")
    assert "moves through phases" in document.text[lifecycle.start : lifecycle.end]


def test_heading_depth_maps_onto_section_levels(tmp_path: Path) -> None:
    document = MarkdownDocsLoader().load(_corpus(tmp_path))[0]

    # Two heading depths become the same two levels the structural chunker expects.
    assert {s.title for s in document.sections_at_level(1)} == {
        "Pod lifecycle",
        "Pod networking",
    }
    assert {s.title for s in document.sections_at_level(2)} == {"Restart policy"}


def test_headings_below_the_third_level_are_not_sections(tmp_path: Path) -> None:
    document = MarkdownDocsLoader().load(_corpus(tmp_path))[0]

    assert all(s.title != "Backoff detail" for s in document.sections)


def test_files_with_only_front_matter_are_skipped(tmp_path: Path) -> None:
    documents = MarkdownDocsLoader().load(_corpus(tmp_path))

    assert all(d.id != "empty.md" for d in documents)


def test_a_missing_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="No corpus directory"):
        MarkdownDocsLoader().load(tmp_path / "absent")


def test_a_directory_with_no_matches_is_reported(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()

    with pytest.raises(CorpusError, match="No files matching"):
        MarkdownDocsLoader().load(tmp_path / "docs")
