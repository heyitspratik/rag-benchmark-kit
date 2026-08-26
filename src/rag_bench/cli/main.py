"""Root ``rag-bench`` command, to which each sub-application is attached."""

from typing import Annotated

import typer

from rag_bench import __version__
from rag_bench.cli import corpus, index

app = typer.Typer(
    name="rag-bench",
    help="Configurable, benchmarked RAG pipeline.",
    add_completion=False,
)
app.add_typer(corpus.app, name="corpus")
app.add_typer(index.app, name="index")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed version and exit."),
    ] = False,
) -> None:
    """Configurable, benchmarked RAG pipeline."""
    if version:
        typer.echo(__version__)
        raise typer.Exit
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit
