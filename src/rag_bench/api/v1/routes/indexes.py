"""Triggering index builds and polling their progress."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Response, status

from rag_bench.api.dependencies import IndexServiceDep, RequiresApiKey
from rag_bench.api.schemas import ErrorResponse, IndexRequest, IndexResponse

router = APIRouter(prefix="/indexes", tags=["indexes"], dependencies=[RequiresApiKey])


@router.post(
    "",
    response_model=IndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an index build",
    responses={
        202: {"model": IndexResponse, "description": "Accepted; poll the Location header."},
        422: {"model": ErrorResponse, "description": "The config is missing or invalid."},
    },
)
def create_index(
    request: IndexRequest,
    service: IndexServiceDep,
    background: BackgroundTasks,
    response: Response,
) -> IndexResponse:
    """Accept an index build and return immediately.

    Ingesting a real corpus takes minutes, far longer than a request should hold open,
    so this is 202 with a Location header rather than a blocking 201. The config is
    validated before accepting, so a typo fails here rather than invisibly later.
    """
    task = service.create(request.config)
    background.add_task(service.run, task.id, recreate=request.recreate)
    response.headers["Location"] = f"/api/v1/indexes/{task.id}"
    return service.get(task.id)


@router.get(
    "/{index_id}",
    response_model=IndexResponse,
    summary="Report the progress of an index build",
    responses={404: {"model": ErrorResponse, "description": "No such build."}},
)
def get_index(index_id: UUID, service: IndexServiceDep) -> IndexResponse:
    """Report one build's current state."""
    return service.get(index_id)
