"""Fixtures shared by the whole suite."""

from collections.abc import Sequence

import pytest

from rag_bench.core import settings as settings_module
from rag_bench.core.interfaces import BaseEmbedder, Vector
from rag_bench.core.registry import register_embedder

#: Registered once for the whole suite, under a name no real config would use, so tests
#: can drive the pipeline through the registry without downloading model weights.
OFFLINE_EMBEDDER = "_test_offline_embedder"
OFFLINE_DIMENSION = 4


@register_embedder(OFFLINE_EMBEDDER)
class OfflineEmbedder(BaseEmbedder):
    """Derives a vector from the text length, so results are deterministic and free."""

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        return OFFLINE_DIMENSION

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)

    def _vector(self, text: str) -> Vector:
        return [float(len(text) % 7), 1.0, 0.0, 0.0]


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
