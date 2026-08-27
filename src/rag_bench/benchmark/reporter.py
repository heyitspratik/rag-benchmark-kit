"""Turning stored results into a table someone can read.

The markdown output is meant to be pasted straight into the README, which is why the
winning row is emphasised and the columns are ordered the way a reader decides: quality
first, then the cost of getting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, select

from rag_bench.core.exceptions import ResourceNotFoundError
from rag_bench.core.logging import get_logger
from rag_bench.db.models import BenchmarkRun, ConfigurationMetrics, RunConfiguration, RunStatus
from rag_bench.db.session import get_engine, session_scope

logger = get_logger(__name__)

#: Metric used to order the table, with fallbacks for a run that skipped RAGAS.
RANKING_METRICS = ("faithfulness", "hit_rate", "mrr")

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("chunker", "Chunker"),
    ("embedder", "Embedder"),
    ("retriever", "Retriever"),
    ("faithfulness", "Faith."),
    ("answer_relevancy", "Ans.Rel."),
    ("context_precision", "Ctx.Prec."),
    ("context_recall", "Ctx.Rec."),
    ("hit_rate", "Hit@k"),
    ("mrr", "MRR"),
    ("abstention_accuracy", "Abstain"),
    ("p50_retrieval_ms", "p50 Ret."),
    ("p95_generation_ms", "p95 Gen."),
    ("estimated_cost_usd", "Cost $"),
)

_LATENCY_COLUMNS = frozenset(
    {"p50_retrieval_ms", "p95_retrieval_ms", "p50_generation_ms", "p95_generation_ms"}
)


@dataclass(frozen=True)
class ConfigurationRow:
    """One configuration's results, flattened for reporting."""

    fingerprint: str
    status: str
    chunker: str
    embedder: str
    retriever: str
    store: str
    generator: str
    metrics: dict[str, Any]
    by_difficulty: dict[str, Any]
    error: str | None = None

    def value(self, column: str) -> Any:
        """Look up a column by name, whether it is a component or a metric."""
        if hasattr(self, column):
            return getattr(self, column)
        return self.metrics.get(column)


@dataclass(frozen=True)
class RunReport:
    """Everything needed to render a report for one run."""

    run_id: UUID
    name: str
    git_sha: str
    git_dirty: bool
    status: str
    eval_set: str
    question_count: int
    llm_provider: str
    llm_model: str
    started_at: datetime | None
    finished_at: datetime | None
    rows: tuple[ConfigurationRow, ...]

    @property
    def ranking_metric(self) -> str:
        """The metric the table is sorted by.

        Falls back down :data:`RANKING_METRICS` so a run without RAGAS still ranks by
        something meaningful rather than producing an arbitrary order.
        """
        for metric in RANKING_METRICS:
            if any(row.value(metric) is not None for row in self.rows):
                return metric
        return RANKING_METRICS[-1]

    def ranked(self) -> list[ConfigurationRow]:
        """Rows best first, with unscored configurations last.

        Ties on the headline metric fall through to the remaining ranking metrics before
        resorting to names. Two configurations often tie on hit rate while differing
        sharply on where in the result list the right chunk landed, and crowning the
        alphabetically-first of those would misreport the winner.
        """
        primary = self.ranking_metric
        order = (primary, *(m for m in RANKING_METRICS if m != primary))

        def key(row: ConfigurationRow) -> tuple[object, ...]:
            scores = tuple(-(row.value(metric) or 0.0) for metric in order)
            return (row.value(primary) is None, *scores, row.chunker, row.embedder, row.retriever)

        return sorted(self.rows, key=key)


def load_report(run_id: UUID, engine: Engine | None = None) -> RunReport:
    """Read one run's results out of the database.

    Args:
        run_id: The run to report on.
        engine: Database engine; the shared one when omitted.

    Returns:
        The assembled report.

    Raises:
        ResourceNotFoundError: If the run does not exist.
    """
    with session_scope(engine or get_engine()) as session:
        run = session.get(BenchmarkRun, run_id)
        if run is None:
            raise ResourceNotFoundError(
                f"No benchmark run with ID {run_id}", details={"run_id": str(run_id)}
            )

        pairs = session.execute(
            select(RunConfiguration, ConfigurationMetrics)
            .outerjoin(
                ConfigurationMetrics,
                ConfigurationMetrics.configuration_id == RunConfiguration.id,
            )
            .where(RunConfiguration.run_id == run_id)
        ).all()

        rows = tuple(
            ConfigurationRow(
                fingerprint=configuration.fingerprint,
                status=str(configuration.status),
                chunker=configuration.chunker,
                embedder=configuration.embedder,
                retriever=configuration.retriever,
                store=configuration.store,
                generator=configuration.generator,
                metrics=_metric_dict(metrics),
                by_difficulty=dict(metrics.by_difficulty) if metrics else {},
                error=configuration.error,
            )
            for configuration, metrics in pairs
        )

        return RunReport(
            run_id=run.id,
            name=run.name,
            git_sha=run.git_sha,
            git_dirty=run.git_dirty,
            status=str(run.status),
            eval_set=run.eval_set,
            question_count=run.question_count,
            llm_provider=run.llm_provider,
            llm_model=run.llm_model,
            started_at=run.started_at,
            finished_at=run.finished_at,
            rows=rows,
        )


def latest_run_id(engine: Engine | None = None) -> UUID:
    """The most recently started run.

    Args:
        engine: Database engine; the shared one when omitted.

    Returns:
        Its ID.

    Raises:
        ResourceNotFoundError: If no run has ever been recorded.
    """
    with session_scope(engine or get_engine()) as session:
        run_id = session.scalar(
            select(BenchmarkRun.id).order_by(BenchmarkRun.created_at.desc()).limit(1)
        )
    if run_id is None:
        raise ResourceNotFoundError("No benchmark runs recorded yet")
    return run_id


def to_markdown(report: RunReport) -> str:
    """Render a report as a markdown table ready to paste into the README.

    Args:
        report: The report to render.

    Returns:
        The markdown document.
    """
    ranked = report.ranked()
    metric = report.ranking_metric
    lines = [
        f"# Benchmark: {report.name}",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Commit: `{report.git_sha[:12]}`{' (working tree dirty)' if report.git_dirty else ''}",
        f"- Eval set: `{report.eval_set}` ({report.question_count} questions)",
        f"- Generated with: `{report.llm_provider}` / `{report.llm_model or 'unrecorded'}`",
        f"- Configurations: {len(ranked)}",
        f"- Sorted by: **{metric}**",
        "",
    ]

    headers = [label for _, label in _COLUMNS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for position, row in enumerate(ranked):
        cells = [_format(key, row.value(key)) for key, _ in _COLUMNS]
        # The winner is emphasised because the table exists to answer one question.
        if position == 0 and row.value(metric) is not None:
            cells = [f"**{cell}**" for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")

    failures = [row for row in ranked if row.status == RunStatus.FAILED]
    if failures:
        lines.extend(["", "## Failed configurations", ""])
        lines.extend(
            f"- `{row.chunker}` / `{row.embedder}` / `{row.retriever}`: {row.error}"
            for row in failures
        )

    breakdown = _difficulty_section(ranked)
    if breakdown:
        lines.extend(["", *breakdown])

    return "\n".join(lines) + "\n"


def to_json(report: RunReport) -> str:
    """Render a report as JSON, for anything that wants the raw numbers.

    Args:
        report: The report to render.

    Returns:
        A pretty-printed JSON document.
    """
    payload = {
        "run_id": str(report.run_id),
        "name": report.name,
        "git_sha": report.git_sha,
        "git_dirty": report.git_dirty,
        "status": report.status,
        "eval_set": report.eval_set,
        "question_count": report.question_count,
        "llm_provider": report.llm_provider,
        "llm_model": report.llm_model,
        "started_at": report.started_at.isoformat() if report.started_at else None,
        "finished_at": report.finished_at.isoformat() if report.finished_at else None,
        "ranking_metric": report.ranking_metric,
        "configurations": [
            {
                "fingerprint": row.fingerprint,
                "status": row.status,
                "chunker": row.chunker,
                "embedder": row.embedder,
                "retriever": row.retriever,
                "store": row.store,
                "generator": row.generator,
                "metrics": row.metrics,
                "by_difficulty": row.by_difficulty,
                "error": row.error,
            }
            for row in report.ranked()
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_report(report: RunReport, directory: Path) -> tuple[Path, Path]:
    """Write both renderings into a results directory.

    Args:
        report: The report to write.
        directory: Directory to create and write into.

    Returns:
        The markdown and JSON paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "results.md"
    json_path = directory / "results.json"
    markdown_path.write_text(to_markdown(report), encoding="utf-8")
    json_path.write_text(to_json(report), encoding="utf-8")
    logger.info("report.written", run_id=str(report.run_id), directory=str(directory))
    return markdown_path, json_path


def _difficulty_section(rows: list[ConfigurationRow]) -> list[str]:
    """Render hit rate per difficulty band for the top configurations.

    A configuration that wins overall but collapses on multi-hop questions is exactly
    the finding a reader needs, and the aggregate table hides it.
    """
    bands = sorted({band for row in rows for band in row.by_difficulty})
    if not bands:
        return []

    lines = ["## Hit rate by difficulty", ""]
    lines.append("| Chunker | Embedder | Retriever | " + " | ".join(bands) + " |")
    lines.append("|" + "|".join(["---"] * (3 + len(bands))) + "|")
    for row in rows:
        cells = [
            _format("hit_rate", (row.by_difficulty.get(band) or {}).get("hit_rate"))
            for band in bands
        ]
        lines.append(
            f"| {row.chunker} | {row.embedder} | {row.retriever} | " + " | ".join(cells) + " |"
        )
    return lines


def _metric_dict(metrics: ConfigurationMetrics | None) -> dict[str, Any]:
    """Flatten a metrics row into a plain mapping."""
    if metrics is None:
        return {}
    return {
        "faithfulness": metrics.faithfulness,
        "answer_relevancy": metrics.answer_relevancy,
        "context_precision": metrics.context_precision,
        "context_recall": metrics.context_recall,
        "hit_rate": metrics.hit_rate,
        "mrr": metrics.mrr,
        "abstention_accuracy": metrics.abstention_accuracy,
        "p50_retrieval_ms": metrics.p50_retrieval_ms,
        "p95_retrieval_ms": metrics.p95_retrieval_ms,
        "p50_generation_ms": metrics.p50_generation_ms,
        "p95_generation_ms": metrics.p95_generation_ms,
        "total_tokens": metrics.total_tokens,
        "estimated_cost_usd": metrics.estimated_cost_usd,
        "question_count": metrics.question_count,
    }


def _format(column: str, value: object) -> str:
    """Render one cell, keeping units out of the numbers themselves."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if not isinstance(value, int | float):
        return str(value)
    if column in _LATENCY_COLUMNS:
        return f"{float(value):.0f}"
    if column == "estimated_cost_usd":
        return f"{float(value):.4f}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
