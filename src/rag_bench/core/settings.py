"""Environment-backed settings.

The project keeps two layers of configuration strictly apart. Anything secret or
machine-specific lives here, in environment variables loaded from ``.env``. Anything that
describes an *experiment*, which chunker, which embedder, which retriever, lives in the
committed YAML files handled by :mod:`rag_bench.core.config`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type LLMProvider = Literal["ollama", "anthropic", "openai"]
type AppEnv = Literal["dev", "prod"]


def _blank_to_none(value: object) -> object:
    """Treat an empty environment variable as unset.

    ``.env.example`` ships keys with empty values as placeholders, and an empty
    ``SecretStr`` is truthy, so without this an unset key would pass validation.
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


type OptionalSecret = Annotated[SecretStr | None, BeforeValidator(_blank_to_none)]

_ENV_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    populate_by_name=True,
)


class LLMSettings(BaseSettings):
    """Which chat model to talk to, and how to reach it.

    Model names are deliberately per-provider rather than one shared ``LLM_MODEL``:
    ``llama3.2:3b`` and ``gpt-4o-mini`` are not interchangeable strings, and sharing one
    variable across providers makes switching provider silently produce a broken model
    name.
    """

    model_config = _ENV_CONFIG

    provider: LLMProvider = Field(default="ollama", validation_alias="LLM_PROVIDER")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    anthropic_api_key: OptionalSecret = None
    anthropic_model: str = "claude-sonnet-5"

    openai_api_key: OptionalSecret = None
    openai_model: str = "gpt-4o-mini"

    temperature: float = Field(default=0.0, ge=0.0, le=2.0, validation_alias="LLM_TEMPERATURE")
    max_output_tokens: int = Field(default=1024, gt=0, validation_alias="LLM_MAX_OUTPUT_TOKENS")
    request_timeout_s: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)

    @property
    def model_name(self) -> str:
        """The model identifier for the selected provider."""
        return {
            "ollama": self.ollama_model,
            "anthropic": self.anthropic_model,
            "openai": self.openai_model,
        }[self.provider]

    @property
    def api_key(self) -> SecretStr | None:
        """The API key for the selected provider, or ``None`` for key-less providers."""
        return {
            "ollama": None,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
        }[self.provider]

    @model_validator(mode="after")
    def _require_provider_credentials(self) -> Self:
        """Fail at load time, with the offending variable named, rather than mid-query."""
        key_variables: dict[str, tuple[str, SecretStr | None]] = {
            "anthropic": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
            "openai": ("OPENAI_API_KEY", self.openai_api_key),
        }
        required = key_variables.get(self.provider)
        if required is not None and required[1] is None:
            raise ValueError(f"{required[0]} is required when LLM_PROVIDER={self.provider}")

        # Ollama is the default precisely because it needs no key; it needs a reachable
        # base URL instead, which differs between host and container.
        if self.provider == "ollama" and not self.ollama_base_url.strip():
            raise ValueError("OLLAMA_BASE_URL is required when LLM_PROVIDER=ollama")
        return self


class DatabaseSettings(BaseSettings):
    """Postgres connection details for benchmark persistence."""

    model_config = _ENV_CONFIG

    dsn: str = Field(
        default="postgresql+psycopg://rag:rag@localhost:5432/rag_bench",
        validation_alias="POSTGRES_DSN",
    )
    echo: bool = Field(default=False, validation_alias="POSTGRES_ECHO")
    pool_size: int = Field(default=5, ge=1)


class QdrantSettings(BaseSettings):
    """Connection details for the default vector store backend."""

    model_config = _ENV_CONFIG

    url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    api_key: OptionalSecret = Field(default=None, validation_alias="QDRANT_API_KEY")
    timeout_s: float = Field(default=30.0, gt=0.0)


class Settings(BaseSettings):
    """Top-level application settings, composed from the per-concern groups above."""

    model_config = _ENV_CONFIG

    app_env: AppEnv = "dev"
    log_level: str = "INFO"
    api_key: OptionalSecret = Field(default=None, validation_alias="API_KEY")

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)

    @property
    def corpus_dir(self) -> Path:
        """Where downloaded corpora are cached."""
        return self.data_dir / "corpus"

    @property
    def eval_dir(self) -> Path:
        """Where committed evaluation sets live."""
        return self.data_dir / "eval"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process.

    Returns:
        The cached settings instance.
    """
    return Settings()
