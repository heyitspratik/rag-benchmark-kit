"""One error envelope for every failure the API can produce.

Registered as exception handlers rather than raised at call sites. A route that raises
``HTTPException`` directly has to restate the envelope each time, and the shapes drift;
here the domain exceptions already carry a code and a status, so the mapping is one
function and the guarantee holds for every route including the ones not yet written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_bench.api.schemas import ErrorDetail, ErrorResponse
from rag_bench.core.exceptions import RagBenchError
from rag_bench.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

#: Codes for failures that do not come from the project's own exception hierarchy.
VALIDATION_ERROR_CODE = "VALIDATION_ERROR"
HTTP_ERROR_CODE = "HTTP_ERROR"
INTERNAL_ERROR_CODE = "INTERNAL_ERROR"


def error_response(
    request: Request,
    *,
    code: str,
    message: str,
    http_status: int,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    """Build the standard error envelope.

    Args:
        request: The request being answered, used to recover its ID.
        code: Stable machine-readable code.
        message: Human-readable explanation.
        http_status: HTTP status to return.
        details: Structured context.

    Returns:
        A JSON response carrying the envelope and the request ID header.
    """
    request_id = getattr(request.state, "request_id", "")
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details or {}, request_id=request_id)
    )
    return JSONResponse(
        status_code=http_status,
        content=payload.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id},
    )


async def handle_project_error(request: Request, exc: Exception) -> JSONResponse:
    """Render a project exception, which already knows its own code and status."""
    error = exc if isinstance(exc, RagBenchError) else RagBenchError(str(exc))
    logger.warning("api.error", code=error.code, message=error.message, path=request.url.path)
    return error_response(
        request,
        code=error.code,
        message=error.message,
        http_status=error.http_status,
        details=error.details,
    )


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Render a request that failed schema validation."""
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    return error_response(
        request,
        code=VALIDATION_ERROR_CODE,
        message="The request body or parameters are invalid.",
        http_status=status.HTTP_422_UNPROCESSABLE_CONTENT,
        details={"errors": _readable(errors)},
    )


async def handle_http_error(request: Request, exc: Exception) -> JSONResponse:
    """Render the framework's own errors, such as a 404 for an unknown path."""
    http_status = (
        exc.status_code
        if isinstance(exc, StarletteHTTPException)
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    detail = exc.detail if isinstance(exc, StarletteHTTPException) else str(exc)
    return error_response(
        request,
        code=HTTP_ERROR_CODE,
        message=str(detail),
        http_status=http_status,
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Render anything that escaped, so the envelope holds for genuine bugs too.

    This is the framework's error boundary rather than a broad ``except`` in business
    logic: the exception is logged with its traceback and reported as an opaque 500,
    because an unexpected error's message may carry internals a caller should not see.
    """
    logger.exception("api.unhandled_error", path=request.url.path, error=type(exc).__name__)
    return error_response(
        request,
        code=INTERNAL_ERROR_CODE,
        message="An unexpected error occurred. The request ID identifies it in the logs.",
        http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach every handler to an application.

    Args:
        app: The application to configure.
    """
    app.add_exception_handler(RagBenchError, handle_project_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_error)
    app.add_exception_handler(Exception, handle_unexpected_error)


def _readable(errors: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    """Flatten pydantic's error list into something a client can display."""
    return [
        {
            "field": ".".join(str(part) for part in _location(error)) or "<body>",
            "message": str(error.get("msg", "")),
        }
        for error in errors
    ]


def _location(error: Mapping[str, object]) -> tuple[object, ...]:
    """The offending field path from one pydantic error."""
    location = error.get("loc")
    return tuple(location) if isinstance(location, list | tuple) else ()
