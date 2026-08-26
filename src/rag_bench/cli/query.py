"""The `rag-bench query` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_bench.cli.output import console, reporting_errors, setup_logging
from rag_bench.core.config import load_pipeline_config
from rag_bench.core.models import Answer
from rag_bench.pipeline.querier import Querier

DEFAULT_CONFIG = Path("configs/default.yaml")

#: Characters of a cited passage to echo, enough to recognise it without flooding a
#: terminal with a whole article.
_PREVIEW_CHARS = 160


def query(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Pipeline config to answer with.")
    ] = DEFAULT_CONFIG,
    k: Annotated[
        int | None, typer.Option("--k", help="Override how many chunks to retrieve.")
    ] = None,
    show_context: Annotated[
        bool, typer.Option("--show-context", help="Print the retrieved passages.")
    ] = False,
) -> None:
    """Answer a question against the built index."""
    setup_logging()
    with reporting_errors():
        querier = Querier(load_pipeline_config(config))
        try:
            answer = querier.answer(question, k)
        finally:
            querier.store.close()
        _render(answer, show_context=show_context)


def _render(answer: Answer, *, show_context: bool) -> None:
    """Print an answer, its sources, and what it cost."""
    console.print()
    console.print(answer.text)
    console.print()

    if answer.citations:
        console.print("[bold]Sources[/bold]")
        for citation in answer.citations:
            refs = ", ".join(citation.section_refs) or citation.chunk_id
            console.print(f"  [{citation.marker}] {refs}")
    elif not answer.abstained:
        console.print("[yellow]The answer cited no sources.[/yellow]")

    if show_context:
        console.print()
        console.print("[bold]Retrieved context[/bold]")
        for context in answer.contexts:
            refs = ", ".join(context.chunk.section_refs) or context.chunk.doc_id
            preview = " ".join(context.chunk.text.split())[:_PREVIEW_CHARS]
            console.print(f"  {context.rank + 1}. ({refs}) score={context.score:.4f}")
            console.print(f"     {preview}...", style="dim")

    console.print()
    console.print(
        f"[dim]retrieval {answer.retrieval_ms:.0f} ms | "
        f"generation {answer.generation_ms:.0f} ms | "
        f"{answer.usage.total_tokens} tokens[/dim]"
    )
