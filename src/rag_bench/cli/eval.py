"""The `rag-bench eval` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_bench.benchmark.evalset import load_eval_set
from rag_bench.benchmark.verify import EvalSetVerifier, VerificationReport
from rag_bench.cli.output import console, key_value_table, reporting_errors, setup_logging
from rag_bench.components import load_components
from rag_bench.core.registry import LOADERS
from rag_bench.core.settings import get_settings

app = typer.Typer(help="Inspect and verify evaluation sets.", no_args_is_help=True)

DEFAULT_EVAL_SET = Path("data/eval/gdpr_ai_act_qa.jsonl")

#: How many weakly supported pairs to name before the list stops being useful.
_MAX_LISTED = 25


@app.command("verify")
def verify(
    eval_set: Annotated[
        Path, typer.Option("--eval-set", "-e", help="The set to check.")
    ] = DEFAULT_EVAL_SET,
    corpus: Annotated[
        str, typer.Option("--corpus", help="Registered loader name for the corpus.")
    ] = "eu_regulations",
    corpus_path: Annotated[
        Path | None, typer.Option("--corpus-path", help="Where the corpus is cached.")
    ] = None,
    min_support: Annotated[
        float,
        typer.Option("--min-support", help="Answer wording that must appear in the cited text."),
    ] = 0.5,
) -> None:
    """Check every citation resolves and every answer is supported by what it cites.

    Exits non-zero when a pair is unusable, so this can gate a commit.
    """
    setup_logging()
    with reporting_errors():
        load_components()
        path = corpus_path or get_settings().corpus_dir / corpus
        documents = LOADERS.create(corpus).load(path)
        loaded = load_eval_set(eval_set)

        report = EvalSetVerifier(documents, min_support=min_support).verify(loaded)
        _render(report)

        if not report.is_clean:
            raise typer.Exit(code=1)


@app.command("stats")
def stats(
    eval_set: Annotated[
        Path, typer.Option("--eval-set", "-e", help="The set to describe.")
    ] = DEFAULT_EVAL_SET,
) -> None:
    """Report the composition of an evaluation set."""
    setup_logging()
    with reporting_errors():
        loaded = load_eval_set(eval_set)
        by_difficulty = loaded.by_difficulty()
        key_value_table(
            f"{eval_set}",
            {
                "questions": len(loaded),
                **{band.value: len(items) for band, items in sorted(by_difficulty.items())},
                "categories": len({q.category for q in loaded if q.category}),
                "distinct sections cited": len({ref for q in loaded for ref in q.source_refs}),
            },
        )


def _render(report: VerificationReport) -> None:
    """Print the outcome, errors first."""
    key_value_table(
        "Verification",
        {
            "questions": report.question_count,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "mean answer support": f"{report.mean_support:.1%}",
        },
    )

    for finding in report.errors:
        console.print(f"[bold red]error[/bold red]   {finding.question_id}: {finding.message}")

    warnings = report.warnings
    for finding in warnings[:_MAX_LISTED]:
        console.print(f"[yellow]review[/yellow]  {finding.question_id}: {finding.message}")
    if len(warnings) > _MAX_LISTED:
        console.print(f"[dim]... and {len(warnings) - _MAX_LISTED} more to review[/dim]")

    if report.is_clean:
        console.print(
            "\n[green]Every citation resolves.[/green] Support scores are lexical only, "
            "so they cannot tell you an answer is a fair reading of the law."
        )
