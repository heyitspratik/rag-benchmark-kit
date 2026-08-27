"""Listing what the pipeline can be configured with."""

from __future__ import annotations

from fastapi import APIRouter

from rag_bench.api.dependencies import ConfigurationServiceDep
from rag_bench.api.schemas import ComponentsResponse

router = APIRouter(prefix="/configurations", tags=["configurations"])


@router.get(
    "",
    response_model=ComponentsResponse,
    summary="List every registered component, by pipeline stage",
)
def list_configurations(service: ConfigurationServiceDep) -> ComponentsResponse:
    """Report the component names a config file may use.

    Read from the registries rather than a hardcoded list, so a newly registered
    component appears here without this route changing.
    """
    return ComponentsResponse(components=service.components())
