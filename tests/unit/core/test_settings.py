import pytest
from pydantic import ValidationError

from rag_bench.core.settings import LLMSettings, Settings


def _llm(**overrides: object) -> LLMSettings:
    return LLMSettings(_env_file=None, **overrides)


def test_ollama_is_the_default_and_needs_no_key() -> None:
    settings = _llm()

    assert settings.provider == "ollama"
    assert settings.api_key is None
    assert settings.model_name == "llama3.2:3b"


@pytest.mark.parametrize(
    ("provider", "variable"),
    [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")],
)
def test_keyed_provider_without_a_key_names_the_missing_variable(
    provider: str, variable: str
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        _llm(provider=provider)

    assert f"{variable} is required when LLM_PROVIDER={provider}" in str(excinfo.value)


def test_blank_key_counts_as_missing() -> None:
    # .env.example ships `ANTHROPIC_API_KEY=` as a placeholder; an empty SecretStr is
    # truthy, so without normalisation this would silently pass validation.
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY is required"):
        _llm(provider="anthropic", anthropic_api_key="   ")


def test_keyed_provider_with_a_key_loads() -> None:
    settings = _llm(provider="anthropic", anthropic_api_key="sk-test")

    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "sk-test"


def test_model_names_are_resolved_per_provider() -> None:
    common = {
        "ollama_model": "llama3.2:3b",
        "anthropic_model": "claude-sonnet-5",
        "openai_model": "gpt-4o-mini",
        "anthropic_api_key": "k",
        "openai_api_key": "k",
    }

    assert _llm(provider="ollama", **common).model_name == "llama3.2:3b"
    assert _llm(provider="anthropic", **common).model_name == "claude-sonnet-5"
    assert _llm(provider="openai", **common).model_name == "gpt-4o-mini"


def test_empty_ollama_base_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="OLLAMA_BASE_URL is required"):
        _llm(provider="ollama", ollama_base_url="  ")


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _llm(provider="cohere")


def test_provider_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    settings = LLMSettings(_env_file=None)

    assert settings.provider == "openai"
    assert settings.model_name == "gpt-4o-mini"


def test_settings_derive_corpus_and_eval_directories() -> None:
    settings = Settings(_env_file=None)

    assert settings.corpus_dir.as_posix() == "data/corpus"
    assert settings.eval_dir.as_posix() == "data/eval"
