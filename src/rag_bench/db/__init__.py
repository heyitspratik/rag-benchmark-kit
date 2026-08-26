"""Persistence: SQLAlchemy models and session handling for benchmark results."""

from rag_bench.db.models import (
    Base,
    BenchmarkRun,
    ConfigurationMetrics,
    QuestionResult,
    RunConfiguration,
    RunStatus,
)
from rag_bench.db.session import (
    DatabaseError,
    build_engine,
    check_database,
    get_engine,
    get_sessionmaker,
    session_scope,
)

__all__ = [
    "Base",
    "BenchmarkRun",
    "ConfigurationMetrics",
    "DatabaseError",
    "QuestionResult",
    "RunConfiguration",
    "RunStatus",
    "build_engine",
    "check_database",
    "get_engine",
    "get_sessionmaker",
    "session_scope",
]
