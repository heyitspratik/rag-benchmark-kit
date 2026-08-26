"""The exception hierarchy for the whole package.

Every failure the application raises deliberately descends from :class:`RagBenchError`.
Each subclass carries a stable machine-readable ``code`` and an HTTP status, which is what
lets the API render one consistent error envelope from an exception handler instead of
scattering ``HTTPException`` raises through the route functions.
"""

from collections.abc import Mapping
from typing import ClassVar


class RagBenchError(Exception):
    """Base class for every error this package raises on purpose."""

    code: ClassVar[str] = "INTERNAL_ERROR"
    http_status: ClassVar[int] = 500

    def __init__(self, message: str, *, details: Mapping[str, object] | None = None) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description, safe to show to an API caller.
            details: Structured context rendered into the API error envelope.
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, object] = dict(details or {})


class ConfigValidationError(RagBenchError):
    """A YAML experiment config or a settings value failed validation."""

    code: ClassVar[str] = "CONFIG_INVALID"
    http_status: ClassVar[int] = 422


class UnknownComponentError(RagBenchError):
    """A config names a component that is not present in the registry."""

    code: ClassVar[str] = "UNKNOWN_COMPONENT"
    http_status: ClassVar[int] = 422


class CorpusError(RagBenchError):
    """The corpus could not be downloaded, found, or parsed."""

    code: ClassVar[str] = "CORPUS_ERROR"
    http_status: ClassVar[int] = 500


class IndexNotReadyError(RagBenchError):
    """A query was issued against a collection that has not been built yet."""

    code: ClassVar[str] = "INDEX_NOT_READY"
    http_status: ClassVar[int] = 409


class VectorStoreError(RagBenchError):
    """The vector store rejected an operation or is unreachable."""

    code: ClassVar[str] = "VECTOR_STORE_ERROR"
    http_status: ClassVar[int] = 502


class LLMProviderError(RagBenchError):
    """The configured LLM provider is misconfigured, unreachable, or failed a call."""

    code: ClassVar[str] = "LLM_PROVIDER_ERROR"
    http_status: ClassVar[int] = 502


class CitationError(RagBenchError):
    """A generated answer cited context that was never retrieved."""

    code: ClassVar[str] = "INVALID_CITATION"
    http_status: ClassVar[int] = 500


class BenchmarkError(RagBenchError):
    """A benchmark run could not be started, resumed, or scored."""

    code: ClassVar[str] = "BENCHMARK_ERROR"
    http_status: ClassVar[int] = 500


class ResourceNotFoundError(RagBenchError):
    """A requested persisted resource does not exist."""

    code: ClassVar[str] = "NOT_FOUND"
    http_status: ClassVar[int] = 404
