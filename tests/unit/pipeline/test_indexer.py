import pytest

from rag_bench.core.config import PipelineConfig
from rag_bench.core.exceptions import UnknownComponentError
from rag_bench.pipeline.indexer import Indexer


def test_build_populates_the_collection(config: PipelineConfig) -> None:
    indexer = Indexer(config)
    try:
        report = indexer.build()

        assert report.documents == 1
        assert report.chunks > 1
        assert indexer.store.count() == report.chunks
    finally:
        indexer.store.close()


def test_the_report_describes_what_was_written(config: PipelineConfig) -> None:
    indexer = Indexer(config)
    try:
        report = indexer.build()

        assert report.collection == "indexer_test"
        assert report.dimension == 4
        assert report.chunks_per_document == {"controllers.md": report.chunks}
        assert report.elapsed_s >= 0.0
    finally:
        indexer.store.close()


def test_stored_chunks_carry_the_sections_they_overlap(config: PipelineConfig) -> None:
    # This annotation is the indexer's own job, not the chunker's, so that strategies
    # blind to document structure still produce scorable chunks.
    indexer = Indexer(config)
    try:
        indexer.build()
        refs = {ref for chunk in indexer.store.iter_chunks() for ref in chunk.section_refs}

        assert any(ref.endswith("#Fees") for ref in refs)
        assert any(ref.endswith("#Refusals") for ref in refs)
    finally:
        indexer.store.close()


def test_rebuilding_replaces_rather_than_accumulates(config: PipelineConfig) -> None:
    indexer = Indexer(config)
    try:
        first = indexer.build()
        second = indexer.build(recreate=True)

        assert indexer.store.count() == first.chunks == second.chunks
    finally:
        indexer.store.close()


def test_building_without_recreate_keeps_existing_points(config: PipelineConfig) -> None:
    indexer = Indexer(config)
    try:
        first = indexer.build()
        indexer.build(recreate=False)

        # Chunk IDs are deterministic, so re-adding the same corpus overwrites in place.
        assert indexer.store.count() == first.chunks
    finally:
        indexer.store.close()


def test_an_unregistered_component_is_reported(config: PipelineConfig) -> None:
    broken = config.with_component("chunker", "does_not_exist")

    with pytest.raises(UnknownComponentError, match="chunker='does_not_exist'"):
        Indexer(broken)


def test_an_unregistered_retriever_does_not_block_indexing(config: PipelineConfig) -> None:
    # Ingestion never touches retrieval, so a config naming an unknown retriever must
    # still be indexable.
    indexer = Indexer(config.with_component("retriever", "does_not_exist"))
    try:
        assert indexer.build().chunks > 0
    finally:
        indexer.store.close()
