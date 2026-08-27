"""The `rag-bench benchmark` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer

from rag_bench.benchmark.ragas_scorer import EmbedderAdapter, RagasScorer
from rag_bench.benchmark.reporter import (
    latest_run_id,
    load_report,
    to_json,
    to_markdown,
    write_report,
)
from rag_bench.benchmark.runner import DEFAULT_PRICING, BenchmarkRunner, RunSummary, resume
from rag_bench.cli.output import console, key_value_table, reporting_errors, setup_logging
from rag_bench.components import load_components
from rag_bench.core.config import SweepConfig, load_pipeline_config, load_sweep_config
from rag_bench.core.llm import build_chat_model
from rag_bench.core.registry import EMBEDDERS
from rag_bench.core.settings import get_settings
from rag_bench.db.session import check_database

app = typer.Typer(help="Run, resume and report benchmark sweeps.", no_args_is_help=True)

DEFAULT_EXPERIMENT = Path("configs/experiments/full_grid.yaml")


@app.command("run")
def run(
    experiment: Annotated[
        Path, typer.Option("--experiment", "-e", help="Sweep config to expand and run.")
    ] = DEFAULT_EXPERIMENT,
    eval_set: Annotated[
        Path | None,
        typer.Option("--eval-set", help="Override the sweep's eval set, for fast iteration."),
    ] = None,
    pricing: Annotated[
        Path, typer.Option("--pricing", help="Token price table for cost estimates.")
    ] = DEFAULT_PRICING,
    ragas: Annotated[
        bool,
        typer.Option(
            "--ragas",
            help="Also compute the LLM-judged RAGAS metrics. Needs `uv sync --extra ragas` "
            "and a judge model, and costs one call per question per configuration.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show the plan and how many indexes it needs, then stop."),
    ] = False,
) -> None:
    """Expand a sweep and run every configuration in it."""
    setup_logging()
    with reporting_errors():
        sweep = load_sweep_config(experiment)
        runner = BenchmarkRunner(
            sweep, pricing=pricing, eval_set=eval_set, scorer=_scorer(ragas, sweep)
        )

        if dry_run:
            _show_plan(runner)
            return

        check_database()
        summary = runner.run()
        _show_summary(summary)


@app.command("plan")
def show_plan(
    experiment: Annotated[
        Path, typer.Option("--experiment", "-e", help="Sweep config to expand.")
    ] = DEFAULT_EXPERIMENT,
    eval_set: Annotated[
        Path | None, typer.Option("--eval-set", help="Override the eval set.")
    ] = None,
) -> None:
    """Show what a sweep would run, without running it."""
    setup_logging()
    with reporting_errors():
        sweep = load_sweep_config(experiment)
        _show_plan(BenchmarkRunner(sweep, eval_set=eval_set, check_provider=False))


@app.command("resume")
def resume_run(
    run_id: Annotated[UUID, typer.Argument(help="The run to continue.")],
    pricing: Annotated[
        Path, typer.Option("--pricing", help="Token price table for cost estimates.")
    ] = DEFAULT_PRICING,
) -> None:
    """Continue an interrupted run, skipping configurations that already finished."""
    setup_logging()
    with reporting_errors():
        check_database()
        _show_summary(resume(run_id, pricing=pricing))


@app.command("report")
def report(
    run_id: Annotated[
        UUID | None, typer.Argument(help="The run to report on. Defaults to the latest.")
    ] = None,
    output_format: Annotated[
        str, typer.Option("--format", "-f", help="markdown or json.")
    ] = "markdown",
    write: Annotated[
        bool, typer.Option("--write", help="Also write results.md and results.json.")
    ] = False,
) -> None:
    """Print a results table for a run."""
    setup_logging()
    with reporting_errors():
        check_database()
        resolved = run_id or latest_run_id()
        rendered = load_report(resolved)

        if output_format == "json":
            console.print_json(to_json(rendered))
        else:
            # Printed raw, not through rich markup, so it can be pasted verbatim.
            typer.echo(to_markdown(rendered))

        if write:
            directory = get_settings().results_dir / rendered.name
            markdown_path, json_path = write_report(rendered, directory)
            console.print(f"[dim]wrote {markdown_path} and {json_path}[/dim]")


def _scorer(enabled: bool, sweep: SweepConfig) -> RagasScorer | None:
    """Build the RAGAS scorer, or return None when the flag is off.

    The judge defaults to the configured provider, which is also what is being measured.
    Point LLM_PROVIDER at a stronger model before trusting these numbers, or the grades
    reflect the same weaknesses twice.

    Args:
        enabled: Whether the caller asked for RAGAS.
        sweep: The sweep, used to find the embedder RAGAS should score with.

    Returns:
        A scorer, or None.
    """
    if not enabled:
        return None
    load_components()
    base = load_pipeline_config(sweep.base_config)
    embedder = EMBEDDERS.create(base.embedder.name, base.embedder.params)
    return RagasScorer(
        judge=build_chat_model(get_settings().llm),
        embeddings=EmbedderAdapter(embedder),
    )


def _show_plan(runner: BenchmarkRunner) -> None:
    """Print the expansion and what it saves."""
    configurations = sum(len(group) for group in runner.groups)
    key_value_table(
        "Benchmark plan",
        {
            "configurations": configurations,
            "indexes to build": len(runner.groups),
            "ingestions saved": configurations - len(runner.groups),
            "questions": len(runner.eval_set),
            "total queries": configurations * len(runner.eval_set),
        },
    )


def _show_summary(summary: RunSummary) -> None:
    """Print what a run did, including the ID needed to resume or report on it."""
    key_value_table(
        f"Run {summary.name}",
        {
            "run id": summary.run_id,
            "completed": summary.completed,
            "failed": summary.failed,
            "skipped": summary.skipped,
            "indexes built": summary.indexes_built,
            "elapsed": f"{summary.elapsed_s:.1f}s",
        },
    )
    console.print(f"[dim]rag-bench benchmark report {summary.run_id}[/dim]")
