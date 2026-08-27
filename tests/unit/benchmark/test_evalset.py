from pathlib import Path

import pytest

from rag_bench.benchmark.evalset import Difficulty, load_eval_set
from rag_bench.core.exceptions import ConfigValidationError

SMOKE = Path("data/eval/smoke.jsonl")

_GOOD = (
    '{"id": "q1", "question": "Q?", "ground_truth": "A", '
    '"source_refs": ["GDPR Art. 5"], "difficulty": "single_hop"}'
)
_NEGATIVE = '{"id": "q2", "question": "Q?", "ground_truth": "A", "difficulty": "negative"}'


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_committed_smoke_set_loads() -> None:
    evalset = load_eval_set(SMOKE)

    assert len(evalset) == 10
    assert evalset.path == SMOKE


def test_the_smoke_set_contains_negatives() -> None:
    # Measuring whether a configuration hallucinates when it should decline is one of
    # the most valuable things the benchmark can show.
    assert len(load_eval_set(SMOKE).negatives) == 2


def test_questions_group_by_difficulty() -> None:
    grouped = load_eval_set(SMOKE).by_difficulty()

    assert set(grouped) == {
        Difficulty.SINGLE_HOP,
        Difficulty.MULTI_HOP,
        Difficulty.DEFINITIONAL,
        Difficulty.NEGATIVE,
    }
    assert sum(len(v) for v in grouped.values()) == 10


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    assert len(load_eval_set(_write(tmp_path, _GOOD, "", _NEGATIVE))) == 2


def test_a_negative_question_with_sources_is_rejected(tmp_path: Path) -> None:
    # The two halves contradict each other, and would corrupt both hit rate and
    # abstention accuracy at once.
    bad = (
        '{"id": "q3", "question": "Q?", "ground_truth": "A", '
        '"source_refs": ["GDPR Art. 5"], "difficulty": "negative"}'
    )

    with pytest.raises(ConfigValidationError, match="must have no source_refs"):
        load_eval_set(_write(tmp_path, bad))


def test_an_answerable_question_without_sources_is_rejected(tmp_path: Path) -> None:
    bad = '{"id": "q4", "question": "Q?", "ground_truth": "A", "difficulty": "single_hop"}'

    with pytest.raises(ConfigValidationError, match="needs at least one source_ref"):
        load_eval_set(_write(tmp_path, bad))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    # Results are keyed by question ID, so duplicates would overwrite each other.
    with pytest.raises(ConfigValidationError, match="duplicate question IDs: q1"):
        load_eval_set(_write(tmp_path, _GOOD, _GOOD))


def test_an_unknown_difficulty_is_rejected(tmp_path: Path) -> None:
    bad = (
        '{"id": "q5", "question": "Q?", "ground_truth": "A", '
        '"source_refs": ["x"], "difficulty": "impossible"}'
    )

    with pytest.raises(ConfigValidationError, match="difficulty"):
        load_eval_set(_write(tmp_path, bad))


def test_malformed_json_names_the_line(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match=r"eval\.jsonl:2 is not valid JSON"):
        load_eval_set(_write(tmp_path, _GOOD, "{not json"))


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    bad = (
        '{"id": "q6", "question": "Q?", "ground_truth": "A", "source_refs": ["x"], '
        '"difficulty": "single_hop", "notes": "typo field"}'
    )

    with pytest.raises(ConfigValidationError, match="notes"):
        load_eval_set(_write(tmp_path, bad))


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError, match="not found"):
        load_eval_set(tmp_path / "absent.jsonl")


def test_an_empty_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")

    with pytest.raises(ConfigValidationError, match="contains no questions"):
        load_eval_set(path)
