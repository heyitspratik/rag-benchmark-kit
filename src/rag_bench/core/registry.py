"""Name-to-implementation registries, one per swappable pipeline stage.

This is the extension point of the whole project. A component is added by writing a class
and decorating it. No edit to the pipeline, the config schema, or the benchmark runner::

    @register_chunker("my_strategy")
    class MyChunker(BaseChunker):
        def chunk(self, document: Document) -> list[Chunk]: ...

Registration happens on import, so every implementation module must be imported by its
package's ``__init__``; :func:`rag_bench.components.load_components` does that eagerly.
"""

from collections.abc import Callable, Mapping

from rag_bench.core.exceptions import UnknownComponentError
from rag_bench.core.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseGenerator,
    BaseLoader,
    BaseRetriever,
    BaseVectorStore,
)


class Registry[T]:
    """A string-keyed registry of implementations for one pipeline stage."""

    def __init__(self, kind: str) -> None:
        """Initialise an empty registry.

        Args:
            kind: Singular stage name used in error messages, e.g. ``"chunker"``.
        """
        self._kind = kind
        # Stored as a plain callable rather than ``type[T]`` so that implementations are
        # free to declare their own explicit, typed constructor signatures.
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a decorator registering a class under a config-facing name.

        Args:
            name: The name used in YAML config, e.g. ``"structural"``.

        Returns:
            A class decorator that returns the class unchanged.

        Raises:
            ValueError: If the name is already taken, which would otherwise let one
                component silently shadow another depending on import order.
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._factories:
                raise ValueError(f"{self._kind} {name!r} is already registered")
            self._factories[name] = cls
            return cls

        return decorator

    def create(self, name: str, params: Mapping[str, object] | None = None) -> T:
        """Instantiate a registered implementation.

        Args:
            name: The registered name.
            params: Keyword arguments from the config's ``params`` block.

        Returns:
            A new instance.

        Raises:
            UnknownComponentError: If nothing is registered under that name.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise UnknownComponentError(
                f"Unknown {self._kind} {name!r}. Available: {', '.join(self.names()) or 'none'}",
                details={"kind": self._kind, "requested": name, "available": self.names()},
            )
        return factory(**(params or {}))

    def names(self) -> list[str]:
        """Every registered name, sorted."""
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        """Whether a name is registered."""
        return name in self._factories


LOADERS: Registry[BaseLoader] = Registry("loader")
CHUNKERS: Registry[BaseChunker] = Registry("chunker")
EMBEDDERS: Registry[BaseEmbedder] = Registry("embedder")
STORES: Registry[BaseVectorStore] = Registry("store")
RETRIEVERS: Registry[BaseRetriever] = Registry("retriever")
GENERATORS: Registry[BaseGenerator] = Registry("generator")

register_loader = LOADERS.register
register_chunker = CHUNKERS.register
register_embedder = EMBEDDERS.register
register_store = STORES.register
register_retriever = RETRIEVERS.register
register_generator = GENERATORS.register


def available_components() -> dict[str, list[str]]:
    """Every registered name grouped by stage, for the CLI and the API.

    Returns:
        Stage name to sorted implementation names.
    """
    return {
        "loader": LOADERS.names(),
        "chunker": CHUNKERS.names(),
        "embedder": EMBEDDERS.names(),
        "store": STORES.names(),
        "retriever": RETRIEVERS.names(),
        "generator": GENERATORS.names(),
    }
