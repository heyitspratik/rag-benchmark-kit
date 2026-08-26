from collections.abc import Iterator
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from rag_bench.core.config import ComponentConfig, PipelineConfig
from rag_bench.core.exceptions import UnknownComponentError
from rag_bench.pipeline.indexer import Indexer
from rag_bench.pipeline.querier import Querier


class StubChatModel:
    """Stands in for a provider so no test needs a key or a running Ollama."""

    def __init__(self, reply: str = "An answer drawn from the context [1].") -> None:
        self.reply = reply
        self.calls = 0

    def invoke(self, messages: list[tuple[str, str]]) -> AIMessage:
        self.calls += 1
        self.messages = messages
        return AIMessage(content=self.reply)


@pytest.fixture
def built(config: PipelineConfig) -> PipelineConfig:
    """Build the index once, then point the config at the query path."""
    indexer = Indexer(config)
    try:
        indexer.build()
    finally:
        indexer.store.close()
    return config.model_copy(
        update={
            "retriever": ComponentConfig(name="dense", params={"top_k": 2}),
            "generator": ComponentConfig(name="cited"),
        }
    )


@pytest.fixture
def chat_model() -> StubChatModel:
    return StubChatModel()


@pytest.fixture
def querier(built: PipelineConfig, chat_model: StubChatModel) -> Iterator[Querier]:
    """A querier whose provider is stubbed at the factory the generator really calls."""
    with (
        patch("rag_bench.pipeline.querier.check_llm_health"),
        patch("rag_bench.core.llm.build_chat_model", return_value=chat_model),
    ):
        instance = Querier(built)
        yield instance
    instance.store.close()


def test_answering_returns_a_cited_answer(querier: Querier) -> None:
    answer = querier.answer("What fee may a controller charge?")

    assert answer.text.startswith("An answer")
    assert len(answer.citations) == 1
    assert answer.contexts


def test_both_stages_are_timed_separately(querier: Querier) -> None:
    answer = querier.answer("What fee may a controller charge?")

    assert answer.retrieval_ms > 0.0
    assert answer.generation_ms > 0.0


def test_the_call_can_override_how_much_is_retrieved(querier: Querier) -> None:
    assert len(querier.answer("a question", k=1).contexts) == 1


def test_an_abstention_is_carried_through(querier: Querier, chat_model: StubChatModel) -> None:
    chat_model.reply = "INSUFFICIENT_CONTEXT"

    answer = querier.answer("What is the capital of France?")

    assert answer.abstained is True
    assert answer.citations == ()


def test_the_retriever_is_reused_across_questions(querier: Querier) -> None:
    # The benchmark asks a hundred questions per configuration; rebuilding retrieval
    # state per question would dominate the measured latency.
    first = querier.retriever

    querier.answer("one")
    querier.answer("two")

    assert querier.retriever is first


def test_the_provider_is_probed_before_any_question_is_asked(built: PipelineConfig) -> None:
    # Discovering the provider is down on question 40 of 100 wastes the whole run.
    with patch("rag_bench.pipeline.querier.check_llm_health") as probe:
        instance = Querier(built)
        instance.store.close()

    probe.assert_called_once()


def test_the_probe_can_be_skipped(built: PipelineConfig) -> None:
    with patch("rag_bench.pipeline.querier.check_llm_health") as probe:
        instance = Querier(built, check_provider=False)
        instance.store.close()

    probe.assert_not_called()


@pytest.mark.parametrize("stage", ["retriever", "generator"])
def test_an_unregistered_component_is_reported(built: PipelineConfig, stage: str) -> None:
    with pytest.raises(UnknownComponentError, match=f"{stage}='does_not_exist'"):
        Querier(built.with_component(stage, "does_not_exist"), check_provider=False)
