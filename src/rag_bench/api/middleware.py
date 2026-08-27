"""Request-scoped context.

Every request carries an ID, either the caller's or one generated here, and that ID is
bound into the logging context for the life of the request. It is what turns "something
failed" in a support message into a log query, which is why the error envelope carries
it back to the caller.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from rag_bench.core.logging import get_logger, log_context

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

#: Paths that answer constantly and would otherwise bury real traffic in the logs.
_QUIET_PATHS = frozenset({"/health/live", "/health/ready"})

_MS_PER_SECOND = 1000.0


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID, binds it to the logs, and records the outcome."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Handle one request.

        Args:
            request: The incoming request.
            call_next: The rest of the application.

        Returns:
            The response, with the request ID echoed in its headers.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()

        with log_context(request_id=request_id):
            response = await call_next(request)

        elapsed_ms = (time.perf_counter() - started) * _MS_PER_SECOND
        response.headers[REQUEST_ID_HEADER] = request_id

        if request.url.path not in _QUIET_PATHS:
            logger.info(
                "api.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(elapsed_ms, 1),
                request_id=request_id,
            )
        return response
