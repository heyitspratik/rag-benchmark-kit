from typing import Any
from unittest.mock import patch

import httpx
import pytest

from rag_bench.core.exceptions import LLMProviderError
from rag_bench.core.llm import build_chat_model, check_llm_health
from rag_bench.core.settings import LLMSettings


def _llm(**overrides: object) -> LLMSettings:
    return LLMSettings(_env_file=None, **overrides)


def test_ollama_model_is_built_from_settings() -> None:
    settings = _llm(ollama_base_url="http://ollama:11434", ollama_model="llama3.2:3b")

    with patch("langchain_ollama.ChatOllama") as chat_ollama:
        build_chat_model(settings)

    kwargs = chat_ollama.call_args.kwargs
    assert kwargs["model"] == "llama3.2:3b"
    assert kwargs["base_url"] == "http://ollama:11434"
    assert "api_key" not in kwargs


def test_anthropic_model_receives_its_own_key_and_model_name() -> None:
    settings = _llm(
        provider="anthropic", anthropic_api_key="sk-a", anthropic_model="claude-sonnet-5"
    )

    with patch("langchain_anthropic.ChatAnthropic") as chat_anthropic:
        build_chat_model(settings)

    kwargs = chat_anthropic.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["anthropic_api_key"].get_secret_value() == "sk-a"


def test_openai_model_receives_its_own_key_and_model_name() -> None:
    settings = _llm(provider="openai", openai_api_key="sk-o", openai_model="gpt-4o-mini")

    with patch("langchain_openai.ChatOpenAI") as chat_openai:
        build_chat_model(settings)

    kwargs = chat_openai.call_args.kwargs
    assert kwargs["model_name"] == "gpt-4o-mini"
    assert kwargs["openai_api_key"].get_secret_value() == "sk-o"


def test_missing_provider_package_becomes_a_provider_error() -> None:
    settings = _llm()

    with (
        patch("langchain_ollama.ChatOllama", side_effect=ImportError("no module")),
        pytest.raises(LLMProviderError, match="not installed"),
    ):
        build_chat_model(settings)


def _tags_response(*names: str) -> httpx.Response:
    payload = {"models": [{"name": name} for name in names]}
    return httpx.Response(200, json=payload, request=httpx.Request("GET", "http://ollama/api/tags"))


def test_health_check_passes_when_the_model_is_pulled() -> None:
    settings = _llm(ollama_model="llama3.2:3b")

    with patch("rag_bench.core.llm.httpx.get", return_value=_tags_response("llama3.2:3b")) as get:
        check_llm_health(settings)

    assert get.call_args.args[0] == "http://localhost:11434/api/tags"


def test_health_check_tolerates_an_implicit_latest_tag() -> None:
    settings = _llm(ollama_model="llama3.2")

    with patch("rag_bench.core.llm.httpx.get", return_value=_tags_response("llama3.2:latest")):
        check_llm_health(settings)


def test_health_check_names_the_pull_command_when_the_model_is_absent() -> None:
    settings = _llm(ollama_model="llama3.2:3b")

    with (
        patch("rag_bench.core.llm.httpx.get", return_value=_tags_response("qwen2:0.5b")),
        pytest.raises(LLMProviderError, match="make pull-models"),
    ):
        check_llm_health(settings)


def test_health_check_reports_an_unreachable_ollama() -> None:
    settings = _llm()

    with (
        patch("rag_bench.core.llm.httpx.get", side_effect=httpx.ConnectError("refused")),
        pytest.raises(LLMProviderError, match="not reachable"),
    ):
        check_llm_health(settings)


def test_health_check_skips_keyed_providers() -> None:
    settings = _llm(provider="anthropic", anthropic_api_key="sk-a")

    with patch("rag_bench.core.llm.httpx.get") as get:
        check_llm_health(settings)

    get.assert_not_called()


@pytest.mark.parametrize("payload", [{}, {"models": "nonsense"}, {"models": [{"id": 1}]}, []])
def test_health_check_survives_an_unexpected_tags_payload(payload: Any) -> None:
    settings = _llm()
    response = httpx.Response(
        200, json=payload, request=httpx.Request("GET", "http://ollama/api/tags")
    )

    with (
        patch("rag_bench.core.llm.httpx.get", return_value=response),
        pytest.raises(LLMProviderError, match="is not pulled"),
    ):
        check_llm_health(settings)
