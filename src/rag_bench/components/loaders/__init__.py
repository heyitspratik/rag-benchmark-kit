"""Corpus loaders. Importing this package registers every implementation."""

from rag_bench.components.loaders.eu_regulations import EuRegulationsLoader
from rag_bench.components.loaders.markdown_docs import MarkdownDocsLoader

__all__ = ["EuRegulationsLoader", "MarkdownDocsLoader"]
