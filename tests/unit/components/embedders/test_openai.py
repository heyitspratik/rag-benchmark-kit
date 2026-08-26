from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag_bench.components.embedders.openai import OpenAIEmbedder
from rag_bench.core.exceptions import ConfigValidationError, LLMProviderError
from rag_bench.core.settings import get_settings


def _client(*batches: list[list[float]]) -> MagicMock:
    client = MagicMock()
    client.embeddings.create.side_effect = [
        SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v) for i, v in enumerate(batch)])
        for batch in batches
    ]
    return client


def _with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()


def test_dimension_is_known_for_the_default_model() -> None:
    assert OpenAIEmbedder().dimension == 1536


def test_an_unfamiliar_model_must_declare_its_dimension() -> None:
    with pytest.raises(ConfigValidationError, match="Unknown vector length"):
        OpenAIEmbedder(model="some-future-model")


def test_an_unfamiliar_model_is_accepted_with_an_explicit_dimension() -> None:
    assert OpenAIEmbedder(model="some-future-model", dimensions=256).dimension == 256


def test_requests_are_split_into_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_key(monkeypatch)
    client = _client([[1.0], [2.0]], [[3.0]])

    with patch("openai.OpenAI", return_value=client):
        vectors = OpenAIEmbedder(batch_size=2).embed_documents(["a", "b", "c"])

    assert vectors == [[1.0], [2.0], [3.0]]
    assert client.embeddings.create.call_count == 2


def test_results_are_reordered_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    # The API does not promise response order, and a silent shuffle would attach every
    # vector to the wrong chunk.
    _with_key(monkeypatch)
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[2.0]),
            SimpleNamespace(index=0, embedding=[1.0]),
        ]
    )

    with patch("openai.OpenAI", return_value=client):
        assert OpenAIEmbedder().embed_documents(["a", "b"]) == [[1.0], [2.0]]


def test_a_missing_key_is_reported_when_the_client_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    with pytest.raises(LLMProviderError, match="OPENAI_API_KEY is required"):
        OpenAIEmbedder().embed_query("a question")


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_non_positive_batch_size_is_rejected(batch_size: int) -> None:
    with pytest.raises(ConfigValidationError, match="batch_size must be positive"):
        OpenAIEmbedder(batch_size=batch_size)
