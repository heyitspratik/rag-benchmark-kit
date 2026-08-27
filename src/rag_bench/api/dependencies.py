"""Injectable dependencies.

Services are held on the application state and handed to routes through ``Depends``, so
a test can replace one without patching module globals and without an HTTP round trip.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import Engine

from rag_bench.api.services import (
    BenchmarkRunService,
    ConfigurationService,
    IndexService,
    QueryService,
)
from rag_bench.core.exceptions import RagBenchError
from rag_bench.core.settings import Settings

API_KEY_HEADER = "X-API-Key"

_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


class UnauthorisedError(RagBenchError):
    """The request did not present the configured API key."""

    code = "UNAUTHORISED"
    http_status = 401


def get_settings_dep(request: Request) -> Settings:
    """The settings this application was built with."""
    settings: Settings = request.app.state.settings
    return settings


def get_engine_dep(request: Request) -> Engine:
    """The database engine this application was built with."""
    engine: Engine = request.app.state.engine
    return engine


def get_query_service(request: Request) -> QueryService:
    """The query service."""
    service: QueryService = request.app.state.query_service
    return service


def get_configuration_service(request: Request) -> ConfigurationService:
    """The configuration service."""
    service: ConfigurationService = request.app.state.configuration_service
    return service


def get_index_service(request: Request) -> IndexService:
    """The index service."""
    service: IndexService = request.app.state.index_service
    return service


def get_benchmark_service(request: Request) -> BenchmarkRunService:
    """The benchmark run service."""
    service: BenchmarkRunService = request.app.state.benchmark_service
    return service


def require_api_key(
    request: Request,
    presented: Annotated[str | None, Security(_api_key_scheme)] = None,
) -> None:
    """Reject the request when an API key is configured and not presented.

    Authentication is optional on purpose: the quickstart must work with no setup, so
    the key is enforced only once someone has set one.

    Args:
        request: The incoming request.
        presented: The value of the API key header, if any.

    Raises:
        UnauthorisedError: If a key is configured and the presented one does not match.
    """
    settings: Settings = request.app.state.settings
    expected = settings.api_key
    if expected is None:
        return
    if presented is None or presented != expected.get_secret_value():
        raise UnauthorisedError(f"A valid {API_KEY_HEADER} header is required.")


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
EngineDep = Annotated[Engine, Depends(get_engine_dep)]
QueryServiceDep = Annotated[QueryService, Depends(get_query_service)]
ConfigurationServiceDep = Annotated[ConfigurationService, Depends(get_configuration_service)]
IndexServiceDep = Annotated[IndexService, Depends(get_index_service)]
BenchmarkServiceDep = Annotated[BenchmarkRunService, Depends(get_benchmark_service)]
RequiresApiKey = Depends(require_api_key)
