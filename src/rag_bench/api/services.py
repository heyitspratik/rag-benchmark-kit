"""The work behind each endpoint.

Routes validate, delegate here, and shape the response. Keeping the logic out of the
route functions is what lets it be tested without an HTTP client, and what stops
request handling and domain behaviour from growing into each other.
"""

from __future__ import annotations

import base64
import binascii
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, select

from rag_bench.api.schemas import (
    BenchmarkRunDetail,
    BenchmarkRunSummary,
    CitationOut,
    ConfigurationOut,
    ContextOut,
    IndexResponse,
    IndexStatus,
    MetricsOut,
    QueryResponse,
)
from rag_bench.components import load_components
from rag_bench.core.config import load_pipeline_config
from rag_bench.core.exceptions import (
    ConfigValidationError,
    RagBenchError,
    ResourceNotFoundError,
)
from rag_bench.core.logging import get_logger
from rag_bench.core.models import Answer
from rag_bench.core.registry import available_components
from rag_bench.db.models import BenchmarkRun, ConfigurationMetrics, RunConfiguration
from rag_bench.db.session import session_scope
from rag_bench.pipeline.indexer import Indexer
from rag_bench.pipeline.querier import Querier

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/default.yaml")

#: Page size when a caller does not ask for one.
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


class QueryService:
    """Answers questions against a built index."""

    def __init__(self, default_config: Path = DEFAULT_CONFIG) -> None:
        """Initialise the service.

        Args:
            default_config: Config used when a request names none.
        """
        self._default_config = default_config

    def answer(
        self,
        question: str,
        *,
        config: Path | None = None,
        top_k: int | None = None,
        include_contexts: bool = False,
    ) -> QueryResponse:
        """Answer one question.

        Args:
            question: The user's question.
            config: Pipeline config to answer with.
            top_k: Override how many chunks to retrieve.
            include_contexts: Whether to return the retrieved passages.

        Returns:
            The answer with its citations.

        Raises:
            ConfigValidationError: If the config is missing or invalid.
            IndexNotReadyError: If no index has been built.
            LLMProviderError: If the provider is unreachable.
        """
        pipeline = load_pipeline_config(config or self._default_config)
        querier = Querier(pipeline)
        try:
            answer = querier.answer(question, top_k)
        finally:
            querier.store.close()
        return _to_query_response(answer, include_contexts=include_contexts)


class ConfigurationService:
    """Reports what the server can be configured with."""

    def components(self) -> dict[str, list[str]]:
        """Every registered implementation name, grouped by pipeline stage.

        Returns:
            Stage name to sorted implementation names.
        """
        load_components()
        return available_components()


@dataclass
class IndexTask:
    """The state of one index build."""

    id: uuid.UUID
    config: str
    status: IndexStatus = IndexStatus.PENDING
    documents: int = 0
    chunks: int = 0
    dimension: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None


class IndexService:
    """Starts index builds and reports their progress.

    Task state is held in memory. Index builds are triggered by an operator rather than
    by end users, and they are cheap to repeat, so the cost of losing status on restart
    is small. It does mean a multi-worker deployment would need a shared store, since a
    task started on one worker is invisible to the others.
    """

    def __init__(self, default_config: Path = DEFAULT_CONFIG) -> None:
        """Initialise the service.

        Args:
            default_config: Config used when a request names none.
        """
        self._default_config = default_config
        self._tasks: dict[uuid.UUID, IndexTask] = {}
        self._lock = threading.Lock()

    def create(self, config: Path | None = None) -> IndexTask:
        """Register a build and return it before any work happens.

        Args:
            config: Pipeline config to build from.

        Returns:
            The pending task, whose ID the caller polls.

        Raises:
            ConfigValidationError: If the config is missing or invalid. Validating here
                rather than in the background is what lets a bad request fail with 422
                instead of being accepted and failing invisibly.
        """
        resolved = config or self._default_config
        load_pipeline_config(resolved)
        task = IndexTask(id=uuid.uuid4(), config=str(resolved))
        with self._lock:
            self._tasks[task.id] = task
        return task

    def run(self, task_id: uuid.UUID, *, recreate: bool = True) -> None:
        """Execute a registered build. Intended to run in the background.

        Args:
            task_id: The task to run.
            recreate: Drop the collection first.
        """
        task = self._get(task_id)
        self._update(task_id, status=IndexStatus.RUNNING)

        indexer = None
        try:
            pipeline = load_pipeline_config(Path(task.config))
            indexer = Indexer(pipeline)
            report = indexer.build(recreate=recreate)
        except RagBenchError as exc:
            logger.warning("api.index_failed", task_id=str(task_id), code=exc.code)
            self._update(
                task_id,
                status=IndexStatus.FAILED,
                error=f"{exc.code}: {exc.message}",
                finished_at=datetime.now(UTC),
            )
            return
        finally:
            if indexer is not None:
                indexer.store.close()

        self._update(
            task_id,
            status=IndexStatus.COMPLETED,
            documents=report.documents,
            chunks=report.chunks,
            dimension=report.dimension,
            elapsed_s=report.elapsed_s,
            finished_at=datetime.now(UTC),
        )

    def get(self, task_id: uuid.UUID) -> IndexResponse:
        """Report one build's state.

        Args:
            task_id: The task to look up.

        Returns:
            Its current state.

        Raises:
            ResourceNotFoundError: If no such task exists on this worker.
        """
        return _to_index_response(self._get(task_id))

    def _get(self, task_id: uuid.UUID) -> IndexTask:
        """Fetch a task or fail.

        Raises:
            ResourceNotFoundError: If no such task exists.
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise ResourceNotFoundError(
                f"No index build with ID {task_id}", details={"index_id": str(task_id)}
            )
        return task

    def _update(self, task_id: uuid.UUID, **changes: object) -> None:
        """Apply changes to a task under the lock."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for name, value in changes.items():
                setattr(task, name, value)


class BenchmarkRunService:
    """Reads persisted benchmark results."""

    def __init__(self, engine: Engine) -> None:
        """Initialise the service.

        Args:
            engine: Database engine to read through.
        """
        self._engine = engine

    def list_runs(
        self, cursor: str | None = None, limit: int = DEFAULT_PAGE_SIZE
    ) -> tuple[list[BenchmarkRunSummary], str | None]:
        """Return one page of runs, newest first.

        Args:
            cursor: Opaque cursor from a previous page.
            limit: Maximum runs to return.

        Returns:
            The page and the cursor for the next one, or ``None`` at the end.

        Raises:
            ConfigValidationError: If the cursor is unreadable.
        """
        size = max(1, min(limit, MAX_PAGE_SIZE))
        statement = select(BenchmarkRun).order_by(
            BenchmarkRun.created_at.desc(), BenchmarkRun.id.desc()
        )
        after = decode_cursor(cursor)
        if after is not None:
            created_at, run_id = after
            # Keyset rather than offset: runs are inserted while a client pages through,
            # and an offset would silently skip or repeat rows when that happens.
            statement = statement.where(
                (BenchmarkRun.created_at < created_at)
                | ((BenchmarkRun.created_at == created_at) & (BenchmarkRun.id < run_id))
            )

        with session_scope(self._engine) as session:
            rows = list(session.scalars(statement.limit(size + 1)).all())
            has_more = len(rows) > size
            page = [_to_run_summary(run) for run in rows[:size]]
            next_cursor = (
                encode_cursor(rows[size - 1].created_at, rows[size - 1].id) if has_more else None
            )
        return page, next_cursor

    def get_run(self, run_id: uuid.UUID) -> BenchmarkRunDetail:
        """Return one run with every configuration it produced.

        Args:
            run_id: The run to fetch.

        Returns:
            The run detail.

        Raises:
            ResourceNotFoundError: If the run does not exist.
        """
        with session_scope(self._engine) as session:
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
                .order_by(RunConfiguration.created_at)
            ).all()
            summary = _to_run_summary(run)
            configurations = [
                _to_configuration(configuration, metrics) for configuration, metrics in pairs
            ]
        return BenchmarkRunDetail(**summary.model_dump(), configurations=configurations)


def encode_cursor(created_at: datetime, run_id: uuid.UUID) -> str:
    """Encode a paging position opaquely, so clients cannot construct one by hand.

    Args:
        created_at: Creation timestamp of the last row on the page.
        run_id: Its ID, which breaks ties between rows created in the same instant.

    Returns:
        A URL-safe cursor.
    """
    raw = f"{created_at.isoformat()}|{run_id}".encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str | None) -> tuple[datetime, uuid.UUID] | None:
    """Decode a cursor produced by :func:`encode_cursor`.

    Args:
        cursor: The cursor, or ``None`` for the first page.

    Returns:
        The position, or ``None`` when no cursor was given.

    Raises:
        ConfigValidationError: If the cursor is malformed.
    """
    if not cursor:
        return None

    try:
        timestamp, _, run_id = base64.urlsafe_b64decode(cursor.encode()).decode().partition("|")
        return datetime.fromisoformat(timestamp), uuid.UUID(run_id)
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise ConfigValidationError(
            "The pagination cursor is not valid. Use the next_cursor from a previous page.",
            details={"cursor": cursor},
        ) from exc


def _to_query_response(answer: Answer, *, include_contexts: bool) -> QueryResponse:
    """Shape a domain answer for the wire."""
    return QueryResponse(
        question=answer.question,
        answer=answer.text,
        abstained=answer.abstained,
        citations=[
            CitationOut(marker=c.marker, chunk_id=c.chunk_id, section_refs=list(c.section_refs))
            for c in answer.citations
        ],
        contexts=[
            ContextOut(
                chunk_id=c.chunk.id,
                doc_id=c.chunk.doc_id,
                text=c.chunk.text,
                score=c.score,
                rank=c.rank,
                section_refs=list(c.chunk.section_refs),
            )
            for c in (answer.contexts if include_contexts else ())
        ],
        retrieval_ms=answer.retrieval_ms,
        generation_ms=answer.generation_ms,
        prompt_tokens=answer.usage.prompt_tokens,
        completion_tokens=answer.usage.completion_tokens,
    )


def _to_index_response(task: IndexTask) -> IndexResponse:
    """Shape an index task for the wire."""
    return IndexResponse(
        id=task.id,
        status=task.status,
        config=task.config,
        documents=task.documents,
        chunks=task.chunks,
        dimension=task.dimension,
        elapsed_s=task.elapsed_s,
        error=task.error,
        created_at=task.created_at,
        finished_at=task.finished_at,
    )


def _to_run_summary(run: BenchmarkRun) -> BenchmarkRunSummary:
    """Shape a run row for the wire."""
    return BenchmarkRunSummary(
        id=run.id,
        name=run.name,
        status=str(run.status),
        git_sha=run.git_sha,
        git_dirty=run.git_dirty,
        llm_provider=run.llm_provider,
        llm_model=run.llm_model,
        eval_set=run.eval_set,
        question_count=run.question_count,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def _to_configuration(
    configuration: RunConfiguration, metrics: ConfigurationMetrics | None
) -> ConfigurationOut:
    """Shape one configuration and its scores for the wire."""
    return ConfigurationOut(
        fingerprint=configuration.fingerprint,
        status=str(configuration.status),
        corpus=configuration.corpus,
        chunker=configuration.chunker,
        embedder=configuration.embedder,
        store=configuration.store,
        retriever=configuration.retriever,
        generator=configuration.generator,
        error=configuration.error,
        metrics=None
        if metrics is None
        else MetricsOut(
            faithfulness=metrics.faithfulness,
            answer_relevancy=metrics.answer_relevancy,
            context_precision=metrics.context_precision,
            context_recall=metrics.context_recall,
            hit_rate=metrics.hit_rate,
            mrr=metrics.mrr,
            abstention_accuracy=metrics.abstention_accuracy,
            p50_retrieval_ms=metrics.p50_retrieval_ms,
            p95_retrieval_ms=metrics.p95_retrieval_ms,
            p50_generation_ms=metrics.p50_generation_ms,
            p95_generation_ms=metrics.p95_generation_ms,
            total_tokens=metrics.total_tokens,
            estimated_cost_usd=metrics.estimated_cost_usd,
            question_count=metrics.question_count,
            by_difficulty=dict(metrics.by_difficulty),
        ),
    )
