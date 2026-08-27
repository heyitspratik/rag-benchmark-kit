"""The verifier is only useful if it fails on bad data, so most of these feed it some."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_bench.benchmark.evalset import EvalQuestion, EvalSet, load_eval_set
from rag_bench.benchmark.verify import EvalSetVerifier, FindingKind
from rag_bench.core.models import Document, DocumentSection

CORPUS = Path("data/corpus/eu_regulations")
EVAL_SET = Path("data/eval/gdpr_ai_act_qa.jsonl")

_ARTICLE_TEXT = (
    "Article 12 Transparent information. The controller shall provide information on "
    "action taken on a request without undue delay and in any event within one month of "
    "receipt of the request. That period may be extended by two further months where "
    "necessary, taking into account the complexity and number of the requests."
)


@pytest.fixture
def document() -> Document:
    return Document(
        id="gdpr",
        title="GDPR",
        source="test",
        text=_ARTICLE_TEXT,
        sections=(
            DocumentSection(ref="GDPR Art. 12", start=0, end=len(_ARTICLE_TEXT), level=1),
            DocumentSection(ref="GDPR Art. 12(3)", start=30, end=len(_ARTICLE_TEXT), level=2),
        ),
    )


def _evalset(*questions: EvalQuestion) -> EvalSet:
    return EvalSet(path=Path("test.jsonl"), questions=questions)


def _question(**overrides: object) -> EvalQuestion:
    fields: dict[str, object] = {
        "id": "q1",
        "question": "How quickly must a controller respond?",
        "ground_truth": "Without undue delay and in any event within one month of receipt.",
        "source_refs": ("GDPR Art. 12(3)",),
        "difficulty": "single_hop",
    }
    return EvalQuestion.model_validate(fields | overrides)


def test_a_supported_answer_passes(document: Document) -> None:
    report = EvalSetVerifier([document]).verify(_evalset(_question()))

    assert report.is_clean
    assert report.support_by_question["q1"] > 0.9


def test_a_citation_that_does_not_resolve_is_an_error(document: Document) -> None:
    # Hit rate is computed by matching retrieved sections against these refs, so a ref
    # that names nothing would make every configuration miss it forever.
    report = EvalSetVerifier([document]).verify(
        _evalset(_question(source_refs=("GDPR Art. 999(1)",)))
    )

    assert not report.is_clean
    assert report.errors[0].kind is FindingKind.UNKNOWN_REF
    assert "GDPR Art. 999(1)" in report.errors[0].message


def test_an_answer_absent_from_its_cited_article_is_flagged(document: Document) -> None:
    # This is the failure that matters most: a fabricated answer would score every
    # configuration against a falsehood.
    invented = _question(
        ground_truth=(
            "The controller must reply within fourteen calendar days and pay statutory "
            "compensation of five hundred euro for every day of delay."
        )
    )

    report = EvalSetVerifier([document]).verify(_evalset(invented))

    assert [f.kind for f in report.warnings] == [FindingKind.WEAK_SUPPORT]
    assert report.support_by_question["q1"] < 0.5


def test_the_support_threshold_is_configurable(document: Document) -> None:
    # Four of the five content words appear in the article, so support is 0.8 and the
    # threshold decides whether that counts.
    partial = _question(ground_truth="undue delay month receipt zzzqqq")

    lenient = EvalSetVerifier([document], min_support=0.5).verify(_evalset(partial))
    strict = EvalSetVerifier([document], min_support=0.9).verify(_evalset(partial))

    assert lenient.warnings == ()
    assert [f.kind for f in strict.warnings] == [FindingKind.WEAK_SUPPORT]


def test_the_schema_owns_the_negative_citation_rule() -> None:
    # Enforced when the file loads, and again when the set is assembled, so the verifier
    # does not repeat it. A negative that cites something can never reach the verifier.
    with pytest.raises(ValidationError, match="must have no source_refs"):
        EvalQuestion.model_validate(
            {
                "id": "q2",
                "question": "Out of scope?",
                "ground_truth": "n/a",
                "source_refs": ["GDPR Art. 12"],
                "difficulty": "negative",
            }
        )


def test_a_negative_the_corpus_can_answer_is_flagged_for_review(document: Document) -> None:
    # If the corpus can answer it, abstention accuracy would punish a configuration for
    # doing the right thing.
    negative = _question(
        id="q3",
        difficulty="negative",
        source_refs=(),
        question="controller information request month receipt",
        ground_truth="Not in the corpus.",
    )

    report = EvalSetVerifier([document]).verify(_evalset(negative))

    assert report.warnings[0].kind is FindingKind.NEGATIVE_MAY_BE_ANSWERABLE


def test_a_genuine_negative_passes(document: Document) -> None:
    negative = _question(
        id="q4",
        difficulty="negative",
        source_refs=(),
        question="What is the corporate income tax rate in Ireland?",
        ground_truth="Not in the corpus.",
    )

    report = EvalSetVerifier([document]).verify(_evalset(negative))

    assert report.is_clean
    assert report.warnings == ()


def test_negatives_are_left_out_of_the_support_average(document: Document) -> None:
    report = EvalSetVerifier([document]).verify(
        _evalset(
            _question(),
            _question(id="q5", difficulty="negative", source_refs=(), ground_truth="n/a"),
        )
    )

    assert set(report.support_by_question) == {"q1"}


def test_the_verifier_knows_every_section_of_the_corpus(document: Document) -> None:
    assert EvalSetVerifier([document]).known_refs == {"GDPR Art. 12", "GDPR Art. 12(3)"}


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not downloaded")
def test_the_committed_eval_set_verifies_against_the_real_corpus() -> None:
    # The whole benchmark rests on this, so it is checked rather than assumed.
    from rag_bench.components import load_components
    from rag_bench.core.registry import LOADERS

    load_components()
    documents = LOADERS.create("eu_regulations").load(CORPUS)

    report = EvalSetVerifier(documents).verify(load_eval_set(EVAL_SET))

    assert report.errors == (), [f.message for f in report.errors]
    assert report.mean_support > 0.8


def test_the_committed_eval_set_has_the_shape_the_spec_asks_for() -> None:
    loaded = load_eval_set(EVAL_SET)
    bands = {band.value: len(items) for band, items in loaded.by_difficulty().items()}

    assert len(loaded) >= 100
    assert len(loaded.negatives) == 10
    assert set(bands) == {"single_hop", "multi_hop", "definitional", "negative"}
