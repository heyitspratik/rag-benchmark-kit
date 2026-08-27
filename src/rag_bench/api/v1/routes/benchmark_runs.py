"""Reading persisted benchmark results."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from rag_bench.api.dependencies import BenchmarkServiceDep
from rag_bench.api.schemas import BenchmarkRunDetail, BenchmarkRunSummary, ErrorResponse, Page
from rag_bench.api.services import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/benchmark-runs", tags=["benchmark-runs"])


@router.get(
    "",
    response_model=Page[BenchmarkRunSummary],
    summary="List benchmark runs, newest first",
    responses={422: {"model": ErrorResponse, "description": "The cursor is malformed."}},
)
def list_benchmark_runs(
    service: BenchmarkServiceDep,
    cursor: Annotated[
        str | None, Query(description="Cursor from a previous page's next_cursor.")
    ] = None,
    limit: Annotated[
        int, Query(ge=1, le=MAX_PAGE_SIZE, description="Maximum runs to return.")
    ] = DEFAULT_PAGE_SIZE,
) -> Page[BenchmarkRunSummary]:
    """Return one page of runs."""
    items, next_cursor = service.list_runs(cursor=cursor, limit=limit)
    return Page[BenchmarkRunSummary](items=items, next_cursor=next_cursor)


@router.get(
    "/{run_id}",
    response_model=BenchmarkRunDetail,
    summary="Report one run with its configurations and metrics",
    responses={404: {"model": ErrorResponse, "description": "No such run."}},
)
def get_benchmark_run(run_id: UUID, service: BenchmarkServiceDep) -> BenchmarkRunDetail:
    """Return one run in full."""
    return service.get_run(run_id)
