from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage

from rag_bench.components.generators.cited import (
    ABSTENTION_MESSAGE,
    ABSTENTION_TOKEN,
    CitedGenerator,
)
from rag_bench.core.exceptions import CitationError, ConfigValidationError, LLMProviderError
from rag_bench.core.models import Chunk, ScoredChunk


class StubChatModel:
    """Returns a canned reply and remembers the prompt it was handed."""

    def __init__(self, reply: object, usage: dict[str, int] | None = None) -> None:
        self.reply = reply
        self.usage = usage
        self.messages: list[tuple[str, str]] = []

    def invoke(self, messages: list[tuple[str, str]]) -> AIMessage:
        self.messages = messages
        return AIMessage(content=self.reply, usage_metadata=self.usage)  # type: ignore[arg-type]


class ExplodingChatModel:
    """Fails the way an unreachable provider does."""

    def invoke(self, messages: list[tuple[str, str]]) -> AIMessage:
        raise OSError("connection refused")


def _contexts(count: int = 3) -> tuple[ScoredChunk, ...]:
    return tuple(
        ScoredChunk(
            chunk=Chunk.create(
                doc_id="gdpr",
                ordinal=i,
                text=f"Passage {i} body text.",
                char_start=0,
                char_end=20,
            ).with_sections((f"GDPR Art. {i + 1}",)),
            score=1.0 - i / 10,
            rank=i,
        )
        for i in range(count)
    )


def _generate(reply: object, contexts: Sequence[ScoredChunk] | None = None, **kwargs: object):
    model = StubChatModel(reply)
    generator = CitedGenerator(chat_model=model, **kwargs)  # type: ignore[arg-type]
    supplied = _contexts() if contexts is None else contexts
    return generator.generate("a question", supplied), model


def test_markers_resolve_to_the_chunks_they_name() -> None:
    answer, _ = _generate("A controller may charge a fee [1], within limits [3].")

    assert [c.marker for c in answer.citations] == ["1", "3"]
    assert answer.citations[0].section_refs == ("GDPR Art. 1",)
    assert answer.citations[1].section_refs == ("GDPR Art. 3",)


def test_repeated_markers_produce_one_citation_each() -> None:
    answer, _ = _generate("First [2]. Second [2]. Third [1].")

    assert [c.marker for c in answer.citations] == ["2", "1"]


def test_grouped_markers_are_parsed_separately() -> None:
    answer, _ = _generate("Both sources agree [1][2].")

    assert [c.marker for c in answer.citations] == ["1", "2"]


def test_citing_a_passage_that_was_never_supplied_raises() -> None:
    # An answer citing something it was not given is a defect, not a formatting quirk.
    # Dropping it silently would let a hallucinated source look like a real one.
    with pytest.raises(CitationError, match=r"cites passage \[9\]"):
        _generate("An invented claim [9].")


def test_a_zero_marker_is_rejected() -> None:
    with pytest.raises(CitationError, match=r"cites passage \[0\]"):
        _generate("Off by one [0].")


def test_an_answer_with_no_markers_is_allowed_but_uncited() -> None:
    answer, _ = _generate("A statement with no citation.")

    assert answer.citations == ()
    assert answer.abstained is False


def test_the_abstention_token_becomes_a_readable_message() -> None:
    answer, _ = _generate(ABSTENTION_TOKEN)

    assert answer.abstained is True
    assert answer.text == ABSTENTION_MESSAGE
    assert answer.citations == ()


def test_abstention_is_detected_even_with_surrounding_text() -> None:
    answer, _ = _generate(f"{ABSTENTION_TOKEN}. Nothing in the passages covers this.")

    assert answer.abstained is True


def test_no_context_abstains_without_calling_the_model() -> None:
    model = StubChatModel("should never be used")

    answer = CitedGenerator(chat_model=model).generate("a question", [])  # type: ignore[arg-type]

    assert answer.abstained is True
    assert model.messages == []


def test_passages_are_numbered_and_labelled_in_the_prompt() -> None:
    _, model = _generate("An answer [1].")

    prompt = model.messages[1][1]
    assert "[1] (GDPR Art. 1)" in prompt
    assert "[3] (GDPR Art. 3)" in prompt
    assert "a question" in prompt


def test_the_system_prompt_names_the_abstention_token() -> None:
    _, model = _generate("An answer [1].")

    assert ABSTENTION_TOKEN in model.messages[0][1]


def test_context_is_truncated_to_the_budget() -> None:
    # A budget smaller than the passages must cost recall, not overflow the model.
    _, model = _generate("An answer [1].", max_context_chars=25)

    prompt = model.messages[1][1]
    assert "[1] (" in prompt
    assert "[2] (" not in prompt


def test_a_marker_beyond_the_truncated_context_still_raises() -> None:
    with pytest.raises(CitationError, match=r"only 1 passages"):
        _generate("Cites a dropped passage [3].", max_context_chars=25)


def test_every_supplied_context_is_recorded_even_when_truncated() -> None:
    # Truncation changes what the model saw, not what retrieval returned, and the
    # benchmark scores retrieval on the latter.
    answer, _ = _generate("An answer [1].", max_context_chars=25)

    assert len(answer.contexts) == 3


def test_token_usage_is_captured() -> None:
    model = StubChatModel(
        "An answer [1].",
        {"input_tokens": 512, "output_tokens": 30, "total_tokens": 542},
    )

    answer = CitedGenerator(chat_model=model).generate("q", _contexts())  # type: ignore[arg-type]

    assert answer.usage.prompt_tokens == 512
    assert answer.usage.completion_tokens == 30
    assert answer.usage.total_tokens == 542


def test_a_provider_without_usage_reporting_yields_zeroes() -> None:
    answer, _ = _generate("An answer [1].")

    assert answer.usage.total_tokens == 0


def test_content_blocks_are_flattened() -> None:
    answer, _ = _generate(
        [{"type": "text", "text": "Part one [1]. "}, {"type": "text", "text": "Part two."}]
    )

    assert answer.text == "Part one [1]. Part two."


def test_a_provider_failure_becomes_a_provider_error() -> None:
    generator = CitedGenerator(chat_model=ExplodingChatModel())  # type: ignore[arg-type]

    with pytest.raises(LLMProviderError, match="chat model call failed"):
        generator.generate("q", _contexts())


@pytest.mark.parametrize("budget", [0, -1])
def test_a_non_positive_context_budget_is_rejected(budget: int) -> None:
    with pytest.raises(ConfigValidationError, match="max_context_chars must be positive"):
        CitedGenerator(max_context_chars=budget)
