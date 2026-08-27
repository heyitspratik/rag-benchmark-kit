"""Checking an evaluation set against the corpus it claims to describe.

Every benchmark number rests on this set being correct, so the claim that an answer is
supported by its cited article is checked rather than asserted. Two failures matter most.
A citation that does not resolve makes hit rate meaningless, because no retrieved chunk
can ever match it. An answer whose content is absent from the article it cites is a
hallucination that would score every configuration against a falsehood.

What this cannot do is judge whether an answer is a fair reading of the law. That needs a
human, and the README says plainly that it has not happened yet.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from rag_bench.benchmark.evalset import EvalQuestion, EvalSet
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Document

logger = get_logger(__name__)

#: Below this share of the answer's content words appearing in the cited text, the pair
#: is flagged for a human to read. It is a smoke alarm, not a proof of correctness.
DEFAULT_MIN_SUPPORT = 0.5

#: A negative question whose wording overlaps a real section this much may in fact be
#: answerable, which would make abstention accuracy measure the wrong thing.
DEFAULT_MAX_NEGATIVE_OVERLAP = 0.75

_WORD_RE = re.compile(r"[a-z][a-z0-9'-]+")

#: Words too common to carry evidence of support either way.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "cannot",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "must",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "only",
        "or",
        "other",
        "out",
        "over",
        "own",
        "same",
        "shall",
        "should",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "under",
        "until",
        "up",
        "upon",
        "was",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "who",
        "whom",
        "with",
        "within",
        "without",
        "would",
        "you",
        "your",
    ]
)


class FindingKind(StrEnum):
    """What is wrong with a pair, or what needs a human eye."""

    UNKNOWN_REF = "unknown_ref"
    WEAK_SUPPORT = "weak_support"
    NEGATIVE_MAY_BE_ANSWERABLE = "negative_may_be_answerable"


@dataclass(frozen=True)
class Finding:
    """One problem found in one question."""

    question_id: str
    kind: FindingKind
    message: str
    support: float | None = None

    @property
    def is_error(self) -> bool:
        """Whether this invalidates the pair rather than merely warranting review."""
        return self.kind is FindingKind.UNKNOWN_REF


@dataclass(frozen=True)
class VerificationReport:
    """The outcome of checking a whole evaluation set."""

    question_count: int
    findings: tuple[Finding, ...]
    support_by_question: dict[str, float]

    @property
    def errors(self) -> tuple[Finding, ...]:
        """Findings that make a pair unusable."""
        return tuple(f for f in self.findings if f.is_error)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Findings a human should read before trusting the set."""
        return tuple(f for f in self.findings if not f.is_error)

    @property
    def is_clean(self) -> bool:
        """Whether nothing invalidating was found."""
        return not self.errors

    @property
    def mean_support(self) -> float:
        """Average share of answer content words found in the cited text."""
        values = list(self.support_by_question.values())
        return sum(values) / len(values) if values else 0.0


class EvalSetVerifier:
    """Checks an evaluation set against the corpus text it cites."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        min_support: float = DEFAULT_MIN_SUPPORT,
        max_negative_overlap: float = DEFAULT_MAX_NEGATIVE_OVERLAP,
    ) -> None:
        """Initialise the verifier.

        Args:
            documents: The loaded corpus the set claims to describe.
            min_support: Share of an answer's content words that must appear in the
                cited text before the pair is accepted without a warning.
            max_negative_overlap: Overlap above which a negative question is flagged as
                possibly answerable after all.
        """
        self._min_support = min_support
        self._max_negative_overlap = max_negative_overlap
        self._section_text: dict[str, str] = {}
        self._section_words: dict[str, frozenset[str]] = {}
        for document in documents:
            for section in document.sections:
                body = document.text[section.start : section.end]
                self._section_text[section.ref] = body
                self._section_words[section.ref] = frozenset(_content_words(body))

    @property
    def known_refs(self) -> frozenset[str]:
        """Every citable section reference in the corpus."""
        return frozenset(self._section_text)

    def verify(self, evalset: EvalSet) -> VerificationReport:
        """Check every question in a set.

        Args:
            evalset: The set to check.

        Returns:
            A report listing everything found, with the per-question support scores.
        """
        findings: list[Finding] = []
        support: dict[str, float] = {}

        for question in evalset:
            if question.is_negative:
                findings.extend(self._check_negative(question))
                continue
            question_findings, score = self._check_answerable(question)
            findings.extend(question_findings)
            support[question.id] = score

        report = VerificationReport(
            question_count=len(evalset),
            findings=tuple(findings),
            support_by_question=support,
        )
        logger.info(
            "evalset.verified",
            questions=report.question_count,
            errors=len(report.errors),
            warnings=len(report.warnings),
            mean_support=round(report.mean_support, 3),
        )
        return report

    def _check_answerable(self, question: EvalQuestion) -> tuple[list[Finding], float]:
        """Check that a question's citations resolve and that they support its answer."""
        findings: list[Finding] = []
        missing = [ref for ref in question.source_refs if ref not in self._section_text]
        for ref in missing:
            findings.append(
                Finding(
                    question_id=question.id,
                    kind=FindingKind.UNKNOWN_REF,
                    message=f"cites {ref!r}, which is not a section of the corpus",
                )
            )

        resolved = [ref for ref in question.source_refs if ref in self._section_words]
        if not resolved:
            return findings, 0.0

        cited_words: set[str] = set()
        for ref in resolved:
            cited_words |= self._section_words[ref]

        answer_words = set(_content_words(question.ground_truth))
        score = len(answer_words & cited_words) / len(answer_words) if answer_words else 0.0

        if score < self._min_support:
            findings.append(
                Finding(
                    question_id=question.id,
                    kind=FindingKind.WEAK_SUPPORT,
                    message=(
                        f"only {score:.0%} of the answer's wording appears in "
                        f"{', '.join(resolved)}; read it before trusting it"
                    ),
                    support=score,
                )
            )
        return findings, score

    def _check_negative(self, question: EvalQuestion) -> list[Finding]:
        """Check that a negative question really is unanswerable from the corpus.

        That a negative cites nothing is not checked here: the schema enforces it when
        the file is loaded, and repeating it would be a branch no valid input can reach.
        """
        question_words = set(_content_words(question.question))
        if not question_words:
            return []

        best_ref, best_overlap = "", 0.0
        for ref, words in self._section_words.items():
            overlap = len(question_words & words) / len(question_words)
            if overlap > best_overlap:
                best_ref, best_overlap = ref, overlap

        if best_overlap >= self._max_negative_overlap:
            return [
                Finding(
                    question_id=question.id,
                    kind=FindingKind.NEGATIVE_MAY_BE_ANSWERABLE,
                    message=(
                        f"{best_overlap:.0%} of its wording appears in {best_ref}; "
                        "confirm the corpus really cannot answer it"
                    ),
                    support=best_overlap,
                )
            ]
        return []


def _content_words(text: str) -> list[str]:
    """Lowercase words that carry meaning, with stopwords and single letters removed."""
    return [word for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS]
