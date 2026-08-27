"""Submitting a question and getting a cited answer."""

from __future__ import annotations

from fastapi import APIRouter, status

from rag_bench.api.dependencies import QueryServiceDep, RequiresApiKey
from rag_bench.api.schemas import ErrorResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="/queries", tags=["queries"], dependencies=[RequiresApiKey])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a question from the indexed corpus",
    responses={
        409: {"model": ErrorResponse, "description": "No index has been built yet."},
        422: {"model": ErrorResponse, "description": "The request or config is invalid."},
        502: {"model": ErrorResponse, "description": "The LLM provider is unreachable."},
    },
)
def create_query(request: QueryRequest, service: QueryServiceDep) -> QueryResponse:
    """Answer a question.

    A query creates no server-side resource, so this returns 200 with the answer rather
    than 201 with a Location header.
    """
    return service.answer(
        request.question,
        config=request.config,
        top_k=request.top_k,
        include_contexts=request.include_contexts,
    )
