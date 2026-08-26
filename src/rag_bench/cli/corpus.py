"""The `rag-bench corpus` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_bench.cli.output import console, key_value_table, reporting_errors, setup_logging
from rag_bench.components.loaders.download import CORPUS_SOURCES, download_corpus
from rag_bench.core.settings import get_settings

app = typer.Typer(help="Download and inspect evaluation corpora.", no_args_is_help=True)


@app.command("download")
def download(
    corpus: Annotated[
        str, typer.Option("--corpus", "-c", help="Corpus to fetch.")
    ] = "eu_regulations",
    destination: Annotated[
        Path | None,
        typer.Option("--dest", help="Where to cache. Defaults to data/corpus/<corpus>."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-download files already cached.")
    ] = False,
) -> None:
    """Fetch a corpus and cache it under data/corpus."""
    setup_logging()
    with reporting_errors():
        target = destination or get_settings().corpus_dir / corpus
        paths = download_corpus(corpus, target, force=force)
        key_value_table(
            f"Corpus {corpus}",
            {
                "destination": target,
                "documents": len(paths),
                "bytes": sum(p.stat().st_size for p in paths),
            },
        )


@app.command("list")
def list_corpora() -> None:
    """List the corpora that can be downloaded."""
    for name, documents in sorted(CORPUS_SOURCES.items()):
        console.print(f"[bold]{name}[/bold]")
        for document in documents:
            console.print(f"  {document.doc_id}: {document.title}")
