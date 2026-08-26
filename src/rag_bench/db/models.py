"""SQLAlchemy models for benchmark persistence.

Two decisions shape this schema.

Component names are stored as their own indexed columns as well as inside the resolved
config JSON. Answering "which chunker won" is the whole point of the benchmark, and it
should be a plain indexed ``GROUP BY``, not a JSON traversal.

Every run records the git commit it was produced by. Benchmark numbers that cannot be
traced back to the code that produced them are not reproducible, and a results table
without that link is an assertion rather than evidence.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Explicit constraint names, so a migration can always refer to a constraint by name
# rather than by whatever the database happened to generate.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# JSONB on Postgres, plain JSON elsewhere, so the unit tests can run on SQLite without a
# second schema definition drifting out of step with the real one.
JsonColumn = JSON().with_variant(postgresql.JSONB(), "postgresql")

_SHA_LENGTH = 40
_NAME_LENGTH = 200
_COMPONENT_LENGTH = 64
_FINGERPRINT_LENGTH = 32
_STATUS_LENGTH = 16


class RunStatus(StrEnum):
    """Lifecycle of a benchmark run or of one configuration within it."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Base(DeclarativeBase):
    """Declarative base carrying the shared metadata and naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: JsonColumn,
        list[str]: JsonColumn,
    }


def _uuid_pk() -> Mapped[uuid.UUID]:
    """A client-generated UUID primary key."""
    return mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


class BenchmarkRun(Base):
    """One execution of a sweep: many configurations, one command."""

    __tablename__ = "benchmark_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(_NAME_LENGTH), index=True)
    git_sha: Mapped[str] = mapped_column(String(_SHA_LENGTH))
    git_dirty: Mapped[bool] = mapped_column(
        default=False,
        doc="True when the working tree had uncommitted changes, so the sha is only "
        "an approximation of what ran.",
    )
    status: Mapped[RunStatus] = mapped_column(String(_STATUS_LENGTH), index=True)
    eval_set: Mapped[str] = mapped_column(String(_NAME_LENGTH))
    question_count: Mapped[int] = mapped_column(Integer, default=0)
    sweep_config: Mapped[dict[str, Any]]
    error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    configurations: Mapped[list[RunConfiguration]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (CheckConstraint("question_count >= 0", name="question_count_non_negative"),)


class RunConfiguration(Base):
    """One point in the sweep grid: a fully resolved pipeline, run over the eval set."""

    __tablename__ = "run_configurations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("benchmark_runs.id", ondelete="CASCADE"), index=True
    )

    fingerprint: Mapped[str] = mapped_column(
        String(_FINGERPRINT_LENGTH),
        doc="Hash of the whole resolved config; how a resumed run recognises finished work.",
    )
    index_fingerprint: Mapped[str] = mapped_column(
        String(_FINGERPRINT_LENGTH),
        index=True,
        doc="Hash of only the stages that determine the index, so configurations sharing "
        "an index can be grouped and ingested once.",
    )

    # Denormalised from resolved_config on purpose: grouping results by component is the
    # question this table exists to answer.
    corpus: Mapped[str] = mapped_column(String(_COMPONENT_LENGTH), index=True)
    chunker: Mapped[str] = mapped_column(String(_COMPONENT_LENGTH), index=True)
    embedder: Mapped[str] = mapped_column(String(_COMPONENT_LENGTH), index=True)
    store: Mapped[str] = mapped_column(String(_COMPONENT_LENGTH), index=True)
    retriever: Mapped[str] = mapped_column(String(_COMPONENT_LENGTH), index=True)
    generator: Mapped[str] = mapped_column(String(_COMPONENT_LENGTH), index=True)

    resolved_config: Mapped[dict[str, Any]]
    status: Mapped[RunStatus] = mapped_column(String(_STATUS_LENGTH), index=True)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    run: Mapped[BenchmarkRun] = relationship(back_populates="configurations")
    question_results: Mapped[list[QuestionResult]] = relationship(
        back_populates="configuration", cascade="all, delete-orphan", passive_deletes=True
    )
    metrics: Mapped[ConfigurationMetrics | None] = relationship(
        back_populates="configuration", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        UniqueConstraint("run_id", "fingerprint"),
        Index("ix_run_configurations_run_status", "run_id", "status"),
    )


class QuestionResult(Base):
    """What one configuration produced for one evaluation question."""

    __tablename__ = "question_results"

    id: Mapped[uuid.UUID] = _uuid_pk()
    configuration_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run_configurations.id", ondelete="CASCADE"), index=True
    )

    question_id: Mapped[str] = mapped_column(String(_NAME_LENGTH))
    difficulty: Mapped[str] = mapped_column(String(_COMPONENT_LENGTH), index=True)
    category: Mapped[str | None] = mapped_column(String(_COMPONENT_LENGTH), default=None)

    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    abstained: Mapped[bool] = mapped_column(default=False, index=True)

    retrieved_chunk_ids: Mapped[list[str]]
    retrieved_section_refs: Mapped[list[str]]
    cited_chunk_ids: Mapped[list[str]]

    retrieval_ms: Mapped[float] = mapped_column(Float, default=0.0)
    generation_ms: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)

    scores: Mapped[dict[str, Any]] = mapped_column(
        JsonColumn,
        default=dict,
        doc="Per-question metric scores, keyed by metric name.",
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    configuration: Mapped[RunConfiguration] = relationship(back_populates="question_results")

    __table_args__ = (
        # One answer per question per configuration, so a resumed run that re-answers a
        # question replaces its result rather than double-counting it.
        UniqueConstraint("configuration_id", "question_id"),
    )


class ConfigurationMetrics(Base):
    """Aggregated scores for one configuration, in aggregate and split by difficulty."""

    __tablename__ = "configuration_metrics"

    id: Mapped[uuid.UUID] = _uuid_pk()
    configuration_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run_configurations.id", ondelete="CASCADE"), unique=True
    )

    # The four RAGAS metrics. Nullable because RAGAS is itself LLM-judged and can fail
    # on an individual configuration without invalidating the cheaper measurements.
    faithfulness: Mapped[float | None] = mapped_column(Float, default=None)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, default=None)
    context_precision: Mapped[float | None] = mapped_column(Float, default=None)
    context_recall: Mapped[float | None] = mapped_column(Float, default=None)

    # Computed directly. Cheap, deterministic, and often more informative.
    hit_rate: Mapped[float | None] = mapped_column(Float, default=None)
    mrr: Mapped[float | None] = mapped_column(Float, default=None)
    abstention_accuracy: Mapped[float | None] = mapped_column(Float, default=None)

    p50_retrieval_ms: Mapped[float | None] = mapped_column(Float, default=None)
    p95_retrieval_ms: Mapped[float | None] = mapped_column(Float, default=None)
    p50_generation_ms: Mapped[float | None] = mapped_column(Float, default=None)
    p95_generation_ms: Mapped[float | None] = mapped_column(Float, default=None)

    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    question_count: Mapped[int] = mapped_column(Integer, default=0)

    by_difficulty: Mapped[dict[str, Any]] = mapped_column(
        JsonColumn,
        default=dict,
        doc="The same metrics recomputed per difficulty band. A configuration that wins "
        "overall but collapses on multi-hop questions is the finding worth reading.",
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    configuration: Mapped[RunConfiguration] = relationship(back_populates="metrics")
