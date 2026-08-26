"""The single place a chat model is constructed, for every supported provider.

Ollama is the committed default because it needs no API key and no signup, so a fresh
clone can run the quickstart at zero cost. It is not simply "OpenAI with a different
environment variable", and the differences are handled here rather than leaking into
callers: there is no key to validate, the base URL differs between host and container,
model names are not portable between providers, and the model must be pulled before first
use, which is worth detecting at startup instead of five minutes into a benchmark run.
"""

from typing import TYPE_CHECKING

import httpx
from pydantic import SecretStr

from rag_bench.core.exceptions import LLMProviderError
from rag_bench.core.logging import get_logger
from rag_bench.core.settings import LLMSettings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)

#: How long the startup reachability probe waits before declaring Ollama down.
HEALTH_TIMEOUT_S = 5.0

#: Ollama's own default tag, appended when a configured model name carries none.
_DEFAULT_OLLAMA_TAG = "latest"

_PULL_HINT = "Start it with `make up`, then pull the model with `make pull-models`."


def build_chat_model(settings: LLMSettings) -> "BaseChatModel":
    """Build the LangChain chat model for the configured provider.

    Provider packages are imported lazily so that installing the project does not force
    every provider's SDK to be importable at startup.

    Args:
        settings: Validated LLM settings; credential rules are enforced on load.

    Returns:
        A ready-to-use chat model.

    Raises:
        LLMProviderError: If the provider package is not installed.
    """
    try:
        match settings.provider:
            case "ollama":
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    model=settings.ollama_model,
                    base_url=settings.ollama_base_url,
                    temperature=settings.temperature,
                    num_predict=settings.max_output_tokens,
                )
            case "anthropic":
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(
                    model=settings.anthropic_model,
                    anthropic_api_key=_required_key(settings),
                    temperature=settings.temperature,
                    max_tokens=settings.max_output_tokens,
                    default_request_timeout=settings.request_timeout_s,
                    max_retries=settings.max_retries,
                )
            case "openai":
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model_name=settings.openai_model,
                    openai_api_key=_required_key(settings),
                    temperature=settings.temperature,
                    max_tokens=settings.max_output_tokens,
                    request_timeout=settings.request_timeout_s,
                    max_retries=settings.max_retries,
                )
    except ImportError as exc:
        raise LLMProviderError(
            f"The {settings.provider} provider package is not installed: {exc}",
            details={"provider": settings.provider},
        ) from exc


def check_llm_health(settings: LLMSettings, *, timeout_s: float = HEALTH_TIMEOUT_S) -> None:
    """Verify the configured provider is usable, failing fast with an actionable message.

    Only Ollama is probed. The keyed providers are validated by their credential rules at
    settings load time, and probing them would mean paying for a request on every start.

    Args:
        settings: Validated LLM settings.
        timeout_s: How long to wait for Ollama to respond.

    Raises:
        LLMProviderError: If Ollama is unreachable or the configured model is not pulled.
    """
    if settings.provider != "ollama":
        return

    tags_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(tags_url, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise LLMProviderError(
            f"Ollama is not reachable at {settings.ollama_base_url}. {_PULL_HINT}",
            details={"provider": "ollama", "base_url": settings.ollama_base_url},
        ) from exc

    installed = _installed_model_names(payload)
    wanted = _qualified_model_name(settings.ollama_model)
    if wanted not in installed:
        raise LLMProviderError(
            f"Ollama model {settings.ollama_model!r} is not pulled. {_PULL_HINT}",
            details={"provider": "ollama", "requested": wanted, "installed": sorted(installed)},
        )
    logger.debug("llm.health_ok", provider="ollama", model=wanted)


def _required_key(settings: LLMSettings) -> SecretStr:
    """Return the selected provider's API key.

    Settings validation already refuses to load a keyed provider without its key; this
    restates the invariant where the type checker can see it.
    """
    key = settings.api_key
    if key is None:
        raise LLMProviderError(
            f"No API key configured for provider {settings.provider!r}",
            details={"provider": settings.provider},
        )
    return key


def _qualified_model_name(model: str) -> str:
    """Append Ollama's implicit ``:latest`` tag so name comparisons are exact."""
    return model if ":" in model else f"{model}:{_DEFAULT_OLLAMA_TAG}"


def _installed_model_names(payload: object) -> set[str]:
    """Extract model names from an ``/api/tags`` response, tolerating shape drift."""
    if not isinstance(payload, dict):
        return set()
    models = payload.get("models")
    if not isinstance(models, list):
        return set()
    return {
        _qualified_model_name(entry["name"])
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
