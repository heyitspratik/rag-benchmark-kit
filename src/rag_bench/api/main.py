"""The FastAPI application.

Built by a factory rather than as a module-level singleton, so a test can construct one
with its own settings and database instead of reaching into globals.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import Engine

from rag_bench import __version__
from rag_bench.api.errors import register_error_handlers
from rag_bench.api.middleware import RequestContextMiddleware
from rag_bench.api.services import (
    DEFAULT_CONFIG,
    BenchmarkRunService,
    ConfigurationService,
    IndexService,
    QueryService,
)
from rag_bench.api.v1 import api_router
from rag_bench.api.v1.routes import health
from rag_bench.components import load_components
from rag_bench.core.logging import configure_logging, get_logger
from rag_bench.core.settings import Settings, get_settings
from rag_bench.db.session import get_engine

logger = get_logger(__name__)

DESCRIPTION = """\
A configurable, benchmarked Retrieval-Augmented Generation pipeline.

Every pipeline stage is chosen by name from a registry, so `GET /api/v1/configurations`
reports what this server can actually be configured with rather than a fixed list.

All failures share one envelope: `{"error": {"code", "message", "details", "request_id"}}`.
The `request_id` is echoed in the `X-Request-ID` header and is what correlates a response
with the server logs.\
"""


def create_app(
    settings: Settings | None = None,
    engine: Engine | None = None,
    default_config: Path = DEFAULT_CONFIG,
) -> FastAPI:
    """Build an application.

    Args:
        settings: Application settings; read from the environment when omitted.
        engine: Database engine; the shared one when omitted.
        default_config: Pipeline config used when a request names none.

    Returns:
        A configured application.
    """
    resolved_settings = settings or get_settings()
    configure_logging(app_env=resolved_settings.app_env, log_level=resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Register components once at startup rather than on the first request."""
        load_components()
        logger.info(
            "api.started",
            version=__version__,
            provider=resolved_settings.llm.provider,
            auth="required" if resolved_settings.api_key else "open",
        )
        yield
        logger.info("api.stopped")

    app = FastAPI(
        title="rag-benchmark-kit",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.settings = resolved_settings
    app.state.engine = engine or get_engine()
    app.state.query_service = QueryService(default_config)
    app.state.configuration_service = ConfigurationService()
    app.state.index_service = IndexService(default_config)
    app.state.benchmark_service = BenchmarkRunService(app.state.engine)

    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(api_router)
    return app


app = create_app()
