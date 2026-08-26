"""The `rag-bench index` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_bench.cli.output import key_value_table, reporting_errors, setup_logging
from rag_bench.core.config import load_pipeline_config
from rag_bench.pipeline.indexer import Indexer

app = typer.Typer(help="Build and inspect vector indexes.", no_args_is_help=True)

_DEFAULT_CONFIG = Path("configs/default.yaml")


@app.command("build")
def build(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Pipeline config to build from.")
    ] = _DEFAULT_CONFIG,
    recreate: Annotated[
        bool,
        typer.Option(
            "--recreate/--no-recreate",
            help="Drop the collection first. Leave on unless adding to an existing index.",
        ),
    ] = True,
) -> None:
    """Ingest the configured corpus into the configured vector store."""
    setup_logging()
    with reporting_errors():
        pipeline = load_pipeline_config(config)
        indexer = Indexer(pipeline)
        try:
            report = indexer.build(recreate=recreate)
        finally:
            indexer.store.close()

        key_value_table(
            "Index built",
            {
                "collection": report.collection,
                "chunker": pipeline.chunker.name,
                "embedder": pipeline.embedder.name,
                "documents": report.documents,
                "chunks": report.chunks,
                "dimension": report.dimension,
                "elapsed": f"{report.elapsed_s:.1f}s",
            },
        )


@app.command("status")
def status(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Pipeline config to inspect.")
    ] = _DEFAULT_CONFIG,
) -> None:
    """Report how many chunks the configured collection currently holds."""
    setup_logging()
    with reporting_errors():
        pipeline = load_pipeline_config(config)
        indexer = Indexer(pipeline)
        try:
            key_value_table(
                "Index status",
                {
                    "collection": pipeline.store.params.get("collection", "unknown"),
                    "chunks": indexer.store.count(),
                },
            )
        finally:
            indexer.store.close()
