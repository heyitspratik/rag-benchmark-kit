from itertools import pairwise
from pathlib import Path

import pytest

from rag_bench.components.loaders.eu_regulations import EuRegulationsLoader
from rag_bench.core.exceptions import CorpusError

from .conftest import build_oj_markup, write_corpus


def _loader() -> EuRegulationsLoader:
    return EuRegulationsLoader(min_articles=2)


def test_articles_and_paragraphs_become_sections(oj_corpus: Path) -> None:
    document = _loader().load(oj_corpus)[0]

    refs = [section.ref for section in document.sections]
    assert "GDPR Art. 1" in refs
    assert "GDPR Art. 1(2)" in refs
    assert "GDPR Art. 3" in refs


def test_article_titles_are_captured(oj_corpus: Path) -> None:
    document = _loader().load(oj_corpus)[0]

    article_one = next(s for s in document.sections if s.ref == "GDPR Art. 1")
    assert article_one.title == "Heading for 1"


def test_section_offsets_point_at_the_real_text(oj_corpus: Path) -> None:
    document = _loader().load(oj_corpus)[0]

    paragraph = next(s for s in document.sections if s.ref == "GDPR Art. 2(2)")
    assert "does not apply to households" in document.text[paragraph.start : paragraph.end]


def test_paragraphs_nest_inside_their_article(oj_corpus: Path) -> None:
    document = _loader().load(oj_corpus)[0]

    article = next(s for s in document.sections if s.ref == "GDPR Art. 1")
    paragraphs = [s for s in document.sections if s.ref.startswith("GDPR Art. 1(")]

    assert paragraphs
    assert all(article.start <= p.start and p.end <= article.end for p in paragraphs)


def test_levels_separate_articles_from_paragraphs(oj_corpus: Path) -> None:
    document = _loader().load(oj_corpus)[0]

    assert len(document.sections_at_level(1)) == 3
    assert {s.ref for s in document.sections_at_level(2)} == {
        "GDPR Art. 1(1)",
        "GDPR Art. 1(2)",
        "GDPR Art. 2(1)",
        "GDPR Art. 2(2)",
    }


def test_top_level_sections_do_not_overlap(oj_corpus: Path) -> None:
    articles = _loader().load(oj_corpus)[0].sections_at_level(1)

    for earlier, later in pairwise(articles):
        assert earlier.end <= later.start


def test_recitals_precede_the_first_article(oj_corpus: Path) -> None:
    document = _loader().load(oj_corpus)[0]

    first_article = document.sections_at_level(1)[0]
    assert "fundamental right" in document.text[: first_article.start]


def test_non_breaking_spaces_are_normalised(oj_corpus: Path) -> None:
    document = _loader().load(oj_corpus)[0]

    assert "\u00a0" not in document.text


def test_too_few_articles_is_treated_as_broken_markup(oj_corpus: Path) -> None:
    # The default threshold guards against the source markup silently changing shape.
    with pytest.raises(CorpusError, match="expected at least"):
        EuRegulationsLoader().load(oj_corpus)


def test_missing_corpus_directory_names_the_download_command(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="rag-bench corpus download"):
        _loader().load(tmp_path / "absent")


def test_manifest_listing_a_missing_file_is_reported(tmp_path: Path) -> None:
    root = write_corpus(tmp_path / "corpus", build_oj_markup({"1": ["a"], "2": ["b"]}))
    (root / "gdpr.xhtml").unlink()

    with pytest.raises(CorpusError, match="missing"):
        _loader().load(root)


def test_markup_without_paragraphs_is_reported(tmp_path: Path) -> None:
    root = write_corpus(tmp_path / "corpus", "<html><body><div>no p tags</div></body></html>")

    with pytest.raises(CorpusError, match="No readable text"):
        _loader().load(root)


def test_empty_manifest_document_list_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "manifest.json").write_text('{"corpus": "eu_regulations", "documents": []}')

    with pytest.raises(CorpusError, match="lists no documents"):
        _loader().load(root)
