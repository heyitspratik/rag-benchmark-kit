"""Version 1 of the HTTP API."""

from fastapi import APIRouter

from rag_bench.api.v1.routes import benchmark_runs, configurations, indexes, queries

#: Everything under /api/v1. Health sits outside the version prefix, because an
#: orchestrator's probe should not have to track API versions.
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(queries.router)
api_router.include_router(configurations.router)
api_router.include_router(indexes.router)
api_router.include_router(benchmark_runs.router)

__all__ = ["api_router"]
