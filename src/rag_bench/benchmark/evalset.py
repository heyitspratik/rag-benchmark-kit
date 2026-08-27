"""Loading and validating the evaluation question set."""

from __future__ import annotations

import json
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_bench.core.exceptions import ConfigValidationError


class Difficulty(StrEnum):
    """How hard a question is, and in what way.

    Reporting broken down by these bands is the point: a configuration that wins overall
    but collapses on multi-hop questions is the finding worth reading.
    """

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    DEFINITIONAL = "definitional"
    NEGATIVE = "negative"


class EvalQuestion(BaseModel):
    """One question, its verified answer, and the sections that support it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    ground_truth: str
    source_refs: tuple[str, ...] = ()
    difficulty: Difficulty
    category: str | None = None

    @property
    def is_negative(self) -> bool:
        """Whether the corpus cannot answer this, so the system should decline."""
        return self.difficulty is Difficulty.NEGATIVE

    @model_validator(mode="after")
    def _check_refs_match_difficulty(self) -> Self:
        """Keep the two halves of a negative question consistent.

        A negative question with sources is a contradiction, and it would silently
        corrupt both hit rate and abstention accuracy.
        """
        if self.is_negative and self.source_refs:
            raise ValueError(
                f"{self.id}: a negative question must have no source_refs, "
                f"got {list(self.source_refs)}"
            )
        if not self.is_negative and not self.source_refs:
            raise ValueError(f"{self.id}: a non-negative question needs at least one source_ref")
        return self


class EvalSet(BaseModel):
    """A whole evaluation set, loaded from JSONL."""

    model_config = ConfigDict(frozen=True)

    path: Path
    questions: tuple[EvalQuestion, ...]

    def __len__(self) -> int:
        """Number of questions in the set."""
        return len(self.questions)

    def __iter__(self) -> Iterator[EvalQuestion]:  # type: ignore[override]
        """Iterate the questions in file order."""
        return iter(self.questions)

    @property
    def negatives(self) -> tuple[EvalQuestion, ...]:
        """The questions the corpus deliberately cannot answer."""
        return tuple(q for q in self.questions if q.is_negative)

    def by_difficulty(self) -> dict[Difficulty, tuple[EvalQuestion, ...]]:
        """Questions grouped into their difficulty bands."""
        grouped: dict[Difficulty, list[EvalQuestion]] = {}
        for question in self.questions:
            grouped.setdefault(question.difficulty, []).append(question)
        return {band: tuple(items) for band, items in grouped.items()}


def load_eval_set(path: Path) -> EvalSet:
    """Load and validate a JSONL evaluation set.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        The validated set, in file order.

    Raises:
        ConfigValidationError: If the file is missing, malformed, empty, or contains
            duplicate question IDs.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigValidationError(
            f"Evaluation set not found: {path}", details={"path": str(path)}
        ) from exc

    questions: list[EvalQuestion] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(EvalQuestion.model_validate(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(
                f"{path}:{number} is not valid JSON: {exc}",
                details={"path": str(path), "line": number},
            ) from exc
        except ValidationError as exc:
            raise ConfigValidationError(
                f"{path}:{number} is not a valid question:\n{_format(exc)}",
                details={"path": str(path), "line": number},
            ) from exc

    if not questions:
        raise ConfigValidationError(f"{path} contains no questions", details={"path": str(path)})

    duplicates = sorted({q.id for q in questions if [x.id for x in questions].count(q.id) > 1})
    if duplicates:
        raise ConfigValidationError(
            f"{path} has duplicate question IDs: {', '.join(duplicates)}. Results are keyed "
            "by ID, so duplicates would overwrite each other.",
            details={"path": str(path), "duplicates": duplicates},
        )

    return EvalSet(path=path, questions=tuple(questions))


def _format(exc: ValidationError) -> str:
    """Render pydantic errors as one readable line each."""
    return "\n".join(
        f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
    )
