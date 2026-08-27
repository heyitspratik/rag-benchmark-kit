"""Request and response models for each resource.

Every endpoint declares its shape here. No route returns a bare dict, so the OpenAPI
document is a complete description of the API rather than an outline of it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

_MAX_QUESTION_CHARS = 2000


class CitationOut(BaseModel):
    """A citation, resolved to the chunk it refers to."""

    model_config = ConfigDict(frozen=True)

    marker: str
    chunk_id: str
    section_refs: list[str] = Field(default_factory=list)


class ContextOut(BaseModel):
    """One retrieved chunk, as returned to a caller."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    section_refs: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """A question to answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1, max_length=_MAX_QUESTION_CHARS)
    config: Path | None = Field(
        default=None, description="Pipeline config to answer with. Defaults to the server's."
    )
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Override how many chunks to retrieve."
    )
    include_contexts: bool = Field(
        default=False, description="Return the retrieved passages alongside the answer."
    )


class QueryResponse(BaseModel):
    """An answer with its citations and timings."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    abstained: bool = Field(
        description="True when the context was insufficient and the system declined."
    )
    citations: list[CitationOut] = Field(default_factory=list)
    contexts: list[ContextOut] = Field(default_factory=list)
    retrieval_ms: float
    generation_ms: float
    prompt_tokens: int
    completion_tokens: int


class ComponentsResponse(BaseModel):
    """Every component name the server can be configured with."""

    model_config = ConfigDict(frozen=True)

    components: dict[str, list[str]] = Field(
        description="Registered implementation names, keyed by pipeline stage."
    )


class IndexStatus(StrEnum):
    """Lifecycle of an index build."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexRequest(BaseModel):
    """A request to build an index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: Path | None = Field(
        default=None, description="Pipeline config to build from. Defaults to the server's."
    )
    recreate: bool = Field(
        default=True,
        description="Drop the collection first. Leave on unless deliberately adding to it.",
    )


class IndexResponse(BaseModel):
    """The state of one index build."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    status: IndexStatus
    config: str
    documents: int = 0
    chunks: int = 0
    dimension: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class MetricsOut(BaseModel):
    """Aggregated scores for one configuration."""

    model_config = ConfigDict(frozen=True)

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    hit_rate: float | None = None
    mrr: float | None = None
    abstention_accuracy: float | None = None
    p50_retrieval_ms: float | None = None
    p95_retrieval_ms: float | None = None
    p50_generation_ms: float | None = None
    p95_generation_ms: float | None = None
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    question_count: int = 0
    by_difficulty: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)


class ConfigurationOut(BaseModel):
    """One configuration within a run, with its scores."""

    model_config = ConfigDict(frozen=True)

    fingerprint: str
    status: str
    corpus: str
    chunker: str
    embedder: str
    store: str
    retriever: str
    generator: str
    error: str | None = None
    metrics: MetricsOut | None = None


class BenchmarkRunSummary(BaseModel):
    """A run as it appears in a list."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    name: str
    status: str
    git_sha: str
    git_dirty: bool
    llm_provider: str
    llm_model: str
    eval_set: str
    question_count: int
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class BenchmarkRunDetail(BenchmarkRunSummary):
    """A run with every configuration it produced."""

    configurations: list[ConfigurationOut] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Liveness or readiness of the service."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="'ok' when every checked dependency answered.")
    checks: dict[str, str] = Field(
        default_factory=dict, description="Per-dependency result, keyed by name."
    )
