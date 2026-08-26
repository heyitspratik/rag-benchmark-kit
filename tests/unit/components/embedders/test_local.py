from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag_bench.components.embedders.local import (
    BGE_QUERY_INSTRUCTION,
    BgeLargeEmbedder,
    BgeSmallEmbedder,
    E5BaseEmbedder,
    SentenceTransformerEmbedder,
)
from rag_bench.core.exceptions import ConfigValidationError


@pytest.fixture
def fake_model() -> MagicMock:
    model = MagicMock()
    model.encode.return_value = [[0.1, 0.2, 0.3]]
    model.get_embedding_dimension.return_value = 384
    return model


def _patched(model: MagicMock) -> Any:
    return patch("sentence_transformers.SentenceTransformer", return_value=model)


def test_bge_applies_its_instruction_to_queries_only(fake_model: MagicMock) -> None:
    fake_model.encode.return_value = [[0.0], [0.0]]

    with _patched(fake_model):
        embedder = BgeSmallEmbedder()
        embedder.embed_query("what is a controller")
        embedder.embed_documents(["a passage", "another passage"])

    assert fake_model.encode.call_args_list[0].args[0] == [
        f"{BGE_QUERY_INSTRUCTION}what is a controller"
    ]
    assert fake_model.encode.call_args_list[1].args[0] == ["a passage", "another passage"]


def test_e5_prefixes_both_sides(fake_model: MagicMock) -> None:
    fake_model.encode.return_value = [[0.0]]

    with _patched(fake_model):
        embedder = E5BaseEmbedder()
        embedder.embed_query("a question")
        embedder.embed_documents(["a passage"])

    assert fake_model.encode.call_args_list[0].args[0] == ["query: a question"]
    assert fake_model.encode.call_args_list[1].args[0] == ["passage: a passage"]


def test_each_model_keeps_its_own_identifier() -> None:
    assert BgeSmallEmbedder.model_id == "BAAI/bge-small-en-v1.5"
    assert BgeLargeEmbedder.model_id == "BAAI/bge-large-en-v1.5"
    assert E5BaseEmbedder.model_id == "intfloat/e5-base-v2"


def test_the_model_is_loaded_once_and_only_on_first_use(fake_model: MagicMock) -> None:
    with _patched(fake_model) as constructor:
        embedder = BgeSmallEmbedder()
        assert constructor.call_count == 0

        embedder.embed_query("first")
        embedder.embed_query("second")

    assert constructor.call_count == 1


def test_dimension_comes_from_the_model(fake_model: MagicMock) -> None:
    with _patched(fake_model):
        assert BgeSmallEmbedder().dimension == 384


def test_a_model_without_a_reported_dimension_is_rejected(fake_model: MagicMock) -> None:
    fake_model.get_embedding_dimension.return_value = None

    with _patched(fake_model), pytest.raises(ConfigValidationError, match="does not report"):
        _ = BgeSmallEmbedder().dimension


def test_batch_size_and_normalisation_reach_the_model(fake_model: MagicMock) -> None:
    with _patched(fake_model):
        BgeSmallEmbedder(batch_size=8, normalize=False).embed_documents(["a"])

    kwargs = fake_model.encode.call_args.kwargs
    assert kwargs["batch_size"] == 8
    assert kwargs["normalize_embeddings"] is False


def test_embedding_no_documents_skips_the_model(fake_model: MagicMock) -> None:
    with _patched(fake_model):
        assert BgeSmallEmbedder().embed_documents([]) == []

    fake_model.encode.assert_not_called()


def test_the_model_identifier_can_be_pinned(fake_model: MagicMock) -> None:
    with _patched(fake_model) as constructor:
        SentenceTransformerEmbedder(model_id="some/other-model").embed_query("x")

    assert constructor.call_args.args[0] == "some/other-model"


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_non_positive_batch_size_is_rejected(batch_size: int) -> None:
    with pytest.raises(ConfigValidationError, match="batch_size must be positive"):
        BgeSmallEmbedder(batch_size=batch_size)
