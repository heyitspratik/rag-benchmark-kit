"""Executing a sweep: build each index once, answer every question, persist everything.

Two properties drive the design.

Indexes are built per group rather than per configuration, so the default grid ingests
the corpus 8 times instead of 24. See :mod:`rag_bench.benchmark.grid`.

Everything is written as it happens, not at the end. A full grid takes hours, and a run
that loses its results because configuration 19 crashed is worthless. That same
incremental write is what makes ``--resume`` possible: finished work is already on disk.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import yaml
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from rag_bench.benchmark.evalset import EvalQuestion, EvalSet, load_eval_set
from rag_bench.benchmark.grid import IndexGroup, plan
from rag_bench.benchmark.metrics import (
    PriceTable,
    QuestionOutcome,
    aggregate,
    aggregate_by_difficulty,
)
from rag_bench.benchmark.ragas_scorer import RagasScorer
from rag_bench.core.config import PipelineConfig, SweepConfig, load_pipeline_config
from rag_bench.core.exceptions import BenchmarkError, RagBenchError, ResourceNotFoundError
from rag_bench.core.logging import get_logger, log_context
from rag_bench.core.models import Answer
from rag_bench.core.settings import Settings, get_settings
from rag_bench.db.models import (
    BenchmarkRun,
    ConfigurationMetrics,
    QuestionResult,
    RunConfiguration,
    RunStatus,
)
from rag_bench.db.session import get_engine, session_scope
from rag_bench.pipeline.indexer import Indexer
from rag_bench.pipeline.querier import Querier

logger = get_logger(__name__)

DEFAULT_PRICING = Path("configs/pricing.yaml")

_GIT_SHA_UNKNOWN = "0" * 40


@dataclass(frozen=True)
class RunSummary:
    """What a completed or partial run produced."""

    run_id: UUID
    name: str
    completed: int
    failed: int
    skipped: int
    indexes_built: int
    elapsed_s: float


class BenchmarkRunner:
    """Runs a sweep end to end and records every result as it is produced."""

    def __init__(
        self,
        sweep: SweepConfig,
        *,
        settings: Settings | None = None,
        engine: Engine | None = None,
        pricing: Path | None = DEFAULT_PRICING,
        eval_set: Path | None = None,
        check_provider: bool = True,
        scorer: RagasScorer | None = None,
    ) -> None:
        """Prepare a run without starting it.

        Args:
            sweep: The sweep declaration.
            settings: Application settings; read from the environment when omitted.
            engine: Database engine; the shared one when omitted.
            pricing: Price table to cost token counts with. ``None`` prices nothing.
            eval_set: Overrides the sweep's own eval set, for a fast smoke run.
            check_provider: Probe the LLM provider before starting.
            scorer: Optional RAGAS scorer. Omit to report only the deterministic
                metrics, which is the default because RAGAS needs a judge model.

        Raises:
            ConfigValidationError: If the base config or eval set is invalid.
        """
        self._sweep = sweep
        self._settings = settings or get_settings()
        self._engine = engine or get_engine()
        self._check_provider = check_provider
        self._scorer = scorer
        self._base = load_pipeline_config(sweep.base_config)
        self._eval: EvalSet = load_eval_set(eval_set or sweep.eval_set)
        self._prices = _load_prices(pricing)
        self._groups = plan(sweep, self._base)

    @property
    def eval_set(self) -> EvalSet:
        """The questions this run will ask."""
        return self._eval

    @property
    def groups(self) -> list[IndexGroup]:
        """The planned index groups."""
        return self._groups

    def start(self) -> UUID:
        """Create the run record and return its ID.

        Returns:
            The new run's ID, which is what ``--resume`` takes.
        """
        sha, dirty = _git_revision()
        run = BenchmarkRun(
            name=self._sweep.name,
            git_sha=sha,
            git_dirty=dirty,
            status=RunStatus.RUNNING,
            eval_set=str(self._eval.path),
            question_count=len(self._eval),
            llm_provider=self._settings.llm.provider,
            llm_model=self._settings.llm.model_name,
            sweep_config=self._sweep.model_dump(mode="json"),
            started_at=datetime.now(UTC),
        )
        with session_scope(self._engine) as session:
            session.add(run)
            session.flush()
            run_id: UUID = run.id
        logger.info("benchmark.started", run_id=str(run_id), name=self._sweep.name, git_sha=sha)
        return run_id

    def execute(self, run_id: UUID) -> RunSummary:
        """Run every configuration that is not already finished.

        Args:
            run_id: The run to execute or resume.

        Returns:
            A summary of what happened.

        Raises:
            ResourceNotFoundError: If the run does not exist.
        """
        started = time.perf_counter()
        self._require_run(run_id)
        done = self._completed_fingerprints(run_id)
        completed = failed = skipped = indexes_built = 0

        with log_context(run_id=str(run_id)):
            for group in self._groups:
                pending = [c for c in group.configurations if c.fingerprint() not in done]
                if not pending:
                    skipped += len(group.configurations)
                    logger.info("benchmark.group_skipped", index=group.fingerprint)
                    continue

                skipped += len(group.configurations) - len(pending)
                # One ingestion serves every retriever variant sharing this index.
                self._build_index(group)
                indexes_built += 1

                for config in pending:
                    if self._run_configuration(run_id, config):
                        completed += 1
                    else:
                        failed += 1

        elapsed = time.perf_counter() - started
        self._finish(run_id, failed)
        summary = RunSummary(
            run_id=run_id,
            name=self._sweep.name,
            completed=completed,
            failed=failed,
            skipped=skipped,
            indexes_built=indexes_built,
            elapsed_s=elapsed,
        )
        logger.info(
            "benchmark.finished",
            run_id=str(run_id),
            completed=completed,
            failed=failed,
            skipped=skipped,
            indexes_built=indexes_built,
            elapsed_s=round(elapsed, 1),
        )
        return summary

    def run(self) -> RunSummary:
        """Start a fresh run and execute it.

        Returns:
            A summary of what happened.
        """
        return self.execute(self.start())

    def _build_index(self, group: IndexGroup) -> None:
        """Ingest the corpus once for a whole group of configurations."""
        indexer = Indexer(group.representative)
        try:
            report = indexer.build(recreate=True)
            logger.info(
                "benchmark.index_built",
                index=group.fingerprint,
                chunks=report.chunks,
                shared_by=len(group),
            )
        finally:
            indexer.store.close()

    def _run_configuration(self, run_id: UUID, config: PipelineConfig) -> bool:
        """Answer every question for one configuration, recording results as they land.

        Returns:
            True when the configuration completed, False when it failed. A failure is
            recorded and the sweep continues: one broken combination should not discard
            the other twenty-three.
        """
        configuration_id = self._register(run_id, config)
        answered: list[QuestionOutcome] = []

        with log_context(configuration=config.fingerprint()):
            try:
                querier = Querier(config, check_provider=self._check_provider)
            except RagBenchError as exc:
                self._fail(configuration_id, exc)
                return False

            try:
                for question in self._eval:
                    outcome = self._answer(configuration_id, querier, question)
                    if outcome is not None:
                        answered.append(outcome)
            finally:
                querier.store.close()

        scored = self._score(configuration_id, answered)
        self._store_metrics(configuration_id, scored)
        self._mark_completed(configuration_id)
        return True

    def _score(
        self, configuration_id: UUID, outcomes: list[QuestionOutcome]
    ) -> list[QuestionOutcome]:
        """Apply the optional LLM-judged metrics, if a scorer was configured.

        A scoring failure downgrades the run to the deterministic metrics rather than
        discarding a configuration's answers, which were expensive to produce.
        """
        if self._scorer is None or not outcomes:
            return outcomes
        try:
            scored = self._scorer.score(outcomes)
        except RagBenchError as exc:
            logger.warning("benchmark.scoring_failed", error=exc.code, message=exc.message)
            return outcomes

        with session_scope(self._engine) as session:
            for outcome in scored:
                row = session.scalar(
                    select(QuestionResult).where(
                        QuestionResult.configuration_id == configuration_id,
                        QuestionResult.question_id == outcome.question.id,
                    )
                )
                if row is not None:
                    row.scores = dict(outcome.scores)
        return scored

    def _answer(
        self, configuration_id: UUID, querier: Querier, question: EvalQuestion
    ) -> QuestionOutcome | None:
        """Answer one question and persist it immediately.

        A question that fails is recorded with its error and skipped, rather than taking
        the whole configuration down with it.
        """
        try:
            answer = querier.answer(question.question)
        except RagBenchError as exc:
            self._store_question_error(configuration_id, question, exc)
            logger.warning("benchmark.question_failed", question_id=question.id, error=exc.code)
            return None

        self._store_question(configuration_id, question, answer)
        return QuestionOutcome(question=question, answer=answer)

    def _register(self, run_id: UUID, config: PipelineConfig) -> UUID:
        """Insert or reuse the configuration row, and mark it running."""
        fingerprint = config.fingerprint()
        with session_scope(self._engine) as session:
            existing = session.scalar(
                select(RunConfiguration).where(
                    RunConfiguration.run_id == run_id,
                    RunConfiguration.fingerprint == fingerprint,
                )
            )
            if existing is None:
                existing = RunConfiguration(
                    run_id=run_id,
                    fingerprint=fingerprint,
                    index_fingerprint=config.index_fingerprint(),
                    corpus=config.corpus.name,
                    chunker=config.chunker.name,
                    embedder=config.embedder.name,
                    store=config.store.name,
                    retriever=config.retriever.name,
                    generator=config.generator.name,
                    resolved_config=config.model_dump(mode="json"),
                    status=RunStatus.RUNNING,
                    started_at=datetime.now(UTC),
                )
                session.add(existing)
                session.flush()
            else:
                existing.status = RunStatus.RUNNING
                existing.started_at = datetime.now(UTC)
                existing.error = None
            configuration_id: UUID = existing.id
        return configuration_id

    def _store_question(
        self, configuration_id: UUID, question: EvalQuestion, answer: Answer
    ) -> None:
        """Persist one answered question, replacing any earlier attempt at it."""
        with session_scope(self._engine) as session:
            self._delete_previous(session, configuration_id, question.id)
            session.add(
                QuestionResult(
                    configuration_id=configuration_id,
                    question_id=question.id,
                    difficulty=question.difficulty.value,
                    category=question.category,
                    question=question.question,
                    answer=answer.text,
                    abstained=answer.abstained,
                    retrieved_chunk_ids=[c.chunk.id for c in answer.contexts],
                    retrieved_section_refs=sorted(
                        {ref for c in answer.contexts for ref in c.chunk.section_refs}
                    ),
                    cited_chunk_ids=[c.chunk_id for c in answer.citations],
                    retrieval_ms=answer.retrieval_ms,
                    generation_ms=answer.generation_ms,
                    prompt_tokens=answer.usage.prompt_tokens,
                    completion_tokens=answer.usage.completion_tokens,
                )
            )

    def _store_question_error(
        self, configuration_id: UUID, question: EvalQuestion, exc: RagBenchError
    ) -> None:
        """Record that one question failed, keeping the rest of the configuration alive."""
        with session_scope(self._engine) as session:
            self._delete_previous(session, configuration_id, question.id)
            session.add(
                QuestionResult(
                    configuration_id=configuration_id,
                    question_id=question.id,
                    difficulty=question.difficulty.value,
                    category=question.category,
                    question=question.question,
                    answer="",
                    retrieved_chunk_ids=[],
                    retrieved_section_refs=[],
                    cited_chunk_ids=[],
                    error=f"{exc.code}: {exc.message}",
                )
            )

    @staticmethod
    def _delete_previous(session: Session, configuration_id: UUID, question_id: str) -> None:
        """Remove an earlier attempt at a question, so a resume replaces rather than clashes.

        The unique constraint on (configuration_id, question_id) would otherwise reject
        the insert when a resumed run re-answers a question it had already attempted.
        """
        previous = session.scalar(
            select(QuestionResult).where(
                QuestionResult.configuration_id == configuration_id,
                QuestionResult.question_id == question_id,
            )
        )
        if previous is not None:
            session.delete(previous)
            session.flush()

    def _store_metrics(self, configuration_id: UUID, outcomes: list[QuestionOutcome]) -> None:
        """Aggregate and persist the metrics for one configuration."""
        model = self._settings.llm.model_name
        summary = aggregate(outcomes, model=model, prices=self._prices)
        by_difficulty = aggregate_by_difficulty(outcomes, model=model, prices=self._prices)

        with session_scope(self._engine) as session:
            existing = session.scalar(
                select(ConfigurationMetrics).where(
                    ConfigurationMetrics.configuration_id == configuration_id
                )
            )
            if existing is not None:
                session.delete(existing)
                session.flush()
            session.add(
                ConfigurationMetrics(
                    configuration_id=configuration_id,
                    faithfulness=summary.faithfulness,
                    answer_relevancy=summary.answer_relevancy,
                    context_precision=summary.context_precision,
                    context_recall=summary.context_recall,
                    hit_rate=summary.hit_rate,
                    mrr=summary.mrr,
                    abstention_accuracy=summary.abstention_accuracy,
                    p50_retrieval_ms=summary.p50_retrieval_ms,
                    p95_retrieval_ms=summary.p95_retrieval_ms,
                    p50_generation_ms=summary.p50_generation_ms,
                    p95_generation_ms=summary.p95_generation_ms,
                    total_tokens=summary.total_tokens,
                    estimated_cost_usd=summary.estimated_cost_usd,
                    question_count=summary.question_count,
                    by_difficulty=by_difficulty,
                )
            )

    def _mark_completed(self, configuration_id: UUID) -> None:
        """Mark a configuration finished, which is what a resume looks for."""
        with session_scope(self._engine) as session:
            configuration = session.get(RunConfiguration, configuration_id)
            if configuration is not None:
                configuration.status = RunStatus.COMPLETED
                configuration.finished_at = datetime.now(UTC)

    def _fail(self, configuration_id: UUID, exc: RagBenchError) -> None:
        """Record that a whole configuration could not run."""
        with session_scope(self._engine) as session:
            configuration = session.get(RunConfiguration, configuration_id)
            if configuration is not None:
                configuration.status = RunStatus.FAILED
                configuration.error = f"{exc.code}: {exc.message}"
                configuration.finished_at = datetime.now(UTC)
        logger.warning("benchmark.configuration_failed", error=exc.code, message=exc.message)

    def _finish(self, run_id: UUID, failed: int) -> None:
        """Close out the run record."""
        with session_scope(self._engine) as session:
            run = session.get(BenchmarkRun, run_id)
            if run is not None:
                run.status = RunStatus.FAILED if failed else RunStatus.COMPLETED
                run.finished_at = datetime.now(UTC)

    def _completed_fingerprints(self, run_id: UUID) -> set[str]:
        """Fingerprints already finished in this run, which a resume skips."""
        with session_scope(self._engine) as session:
            return set(
                session.scalars(
                    select(RunConfiguration.fingerprint).where(
                        RunConfiguration.run_id == run_id,
                        RunConfiguration.status == RunStatus.COMPLETED,
                    )
                ).all()
            )

    def _require_run(self, run_id: UUID) -> None:
        """Fail early when asked to resume something that does not exist."""
        with session_scope(self._engine) as session:
            if session.get(BenchmarkRun, run_id) is None:
                raise ResourceNotFoundError(
                    f"No benchmark run with ID {run_id}", details={"run_id": str(run_id)}
                )


def resume(
    run_id: UUID,
    *,
    engine: Engine | None = None,
    settings: Settings | None = None,
    pricing: Path | None = DEFAULT_PRICING,
    check_provider: bool = True,
) -> RunSummary:
    """Continue an interrupted run from its stored sweep configuration.

    The sweep is rebuilt from what was persisted rather than re-read from disk, so a
    resumed run measures the same grid even if the YAML has since been edited.

    Args:
        run_id: The run to continue.
        engine: Database engine; the shared one when omitted.
        settings: Application settings; read from the environment when omitted.
        pricing: Price table to cost token counts with.
        check_provider: Probe the LLM provider before starting.

    Returns:
        A summary of the work this call did.

    Raises:
        ResourceNotFoundError: If the run does not exist.
        BenchmarkError: If the stored sweep config cannot be rebuilt.
    """
    resolved_engine = engine or get_engine()
    with session_scope(resolved_engine) as session:
        run = session.get(BenchmarkRun, run_id)
        if run is None:
            raise ResourceNotFoundError(
                f"No benchmark run with ID {run_id}", details={"run_id": str(run_id)}
            )
        stored = dict(run.sweep_config)
        eval_set = Path(run.eval_set)

    try:
        sweep = SweepConfig.model_validate(stored)
    except ValueError as exc:
        raise BenchmarkError(
            f"Run {run_id} has an unreadable sweep config: {exc}",
            details={"run_id": str(run_id)},
        ) from exc

    runner = BenchmarkRunner(
        sweep,
        settings=settings,
        engine=resolved_engine,
        pricing=pricing,
        eval_set=eval_set,
        check_provider=check_provider,
    )
    return runner.execute(run_id)


def _load_prices(path: Path | None) -> PriceTable:
    """Read the price table, treating a missing file as everything being free."""
    if path is None or not path.exists():
        return PriceTable()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = raw.get("models") if isinstance(raw, dict) else None
    return PriceTable(models if isinstance(models, dict) else None)


def _git_revision() -> tuple[str, bool]:
    """The commit the run was produced by, and whether the tree was dirty.

    Returns:
        The sha and a dirty flag. A repository that cannot be read yields a zero sha,
        which is honest about the result being untraceable rather than silently wrong.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return _GIT_SHA_UNKNOWN, True
    return sha or _GIT_SHA_UNKNOWN, dirty
