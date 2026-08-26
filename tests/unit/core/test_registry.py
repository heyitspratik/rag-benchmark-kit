import pytest

from rag_bench.core.exceptions import UnknownComponentError
from rag_bench.core.interfaces import BaseChunker
from rag_bench.core.models import Chunk, Document
from rag_bench.core.registry import Registry, available_components


class _Stub(BaseChunker):
    def __init__(self, size: int = 10) -> None:
        self.size = size

    def chunk(self, document: Document) -> list[Chunk]:
        return []


@pytest.fixture
def registry() -> Registry[BaseChunker]:
    return Registry("chunker")


def test_register_then_create_passes_params(registry: Registry[BaseChunker]) -> None:
    registry.register("stub")(_Stub)

    instance = registry.create("stub", {"size": 42})

    assert isinstance(instance, _Stub)
    assert instance.size == 42


def test_create_without_params_uses_defaults(registry: Registry[BaseChunker]) -> None:
    registry.register("stub")(_Stub)

    assert registry.create("stub").size == 10  # type: ignore[attr-defined]


def test_register_returns_the_class_unchanged(registry: Registry[BaseChunker]) -> None:
    assert registry.register("stub")(_Stub) is _Stub


def test_duplicate_registration_is_rejected(registry: Registry[BaseChunker]) -> None:
    registry.register("stub")(_Stub)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("stub")(_Stub)


def test_unknown_name_lists_what_is_available(registry: Registry[BaseChunker]) -> None:
    registry.register("stub")(_Stub)

    with pytest.raises(UnknownComponentError) as excinfo:
        registry.create("nope")

    assert "Unknown chunker 'nope'" in str(excinfo.value)
    assert excinfo.value.details["available"] == ["stub"]


def test_unknown_name_on_empty_registry_says_none(registry: Registry[BaseChunker]) -> None:
    with pytest.raises(UnknownComponentError, match="Available: none"):
        registry.create("nope")


def test_names_are_sorted_and_membership_works(registry: Registry[BaseChunker]) -> None:
    registry.register("zulu")(_Stub)
    registry.register("alpha")(_Stub)

    assert registry.names() == ["alpha", "zulu"]
    assert "alpha" in registry
    assert "missing" not in registry


def test_available_components_covers_every_stage() -> None:
    assert set(available_components()) == {
        "loader",
        "chunker",
        "embedder",
        "store",
        "retriever",
        "generator",
    }
