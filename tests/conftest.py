"""Fixtures shared by the whole suite."""

import pytest

from rag_bench.core import settings as settings_module

# Settings read from the process environment and from a developer's local .env; both are
# cleared so a test's outcome never depends on the machine it runs on.
_ENV_VARS = (
    "APP_ENV",
    "LOG_LEVEL",
    "API_KEY",
    "LLM_PROVIDER",
    "LLM_TEMPERATURE",
    "LLM_MAX_OUTPUT_TOKENS",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "POSTGRES_DSN",
    "QDRANT_URL",
    "QDRANT_API_KEY",
)


_SETTINGS_CLASSES = (
    settings_module.LLMSettings,
    settings_module.DatabaseSettings,
    settings_module.QdrantSettings,
    settings_module.Settings,
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Nested settings groups build themselves, so passing `_env_file=None` to the parent
    # would not reach them; detach the file at the class level instead.
    for cls in _SETTINGS_CLASSES:
        monkeypatch.setitem(cls.model_config, "env_file", None)
    settings_module.get_settings.cache_clear()
