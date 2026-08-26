"""Concrete implementations of every swappable pipeline stage.

Registration happens as a side effect of import, so anything that resolves a component by
name must import this package first. Call :func:`load_components` to say so explicitly.
"""

from rag_bench.components import (
    chunkers,
    embedders,
    generators,
    loaders,
    retrievers,
    stores,
)

__all__ = [
    "chunkers",
    "embedders",
    "generators",
    "load_components",
    "loaders",
    "retrievers",
    "stores",
]


def load_components() -> None:
    """Ensure every implementation module has been imported and registered.

    Importing this package is what does the work. This function exists so a call site can
    state that dependency without an import that reads as unused.
    """
