"""structlog configuration: JSON in production, human-readable in development.

Context such as ``run_id`` or ``request_id`` is bound once, into context variables, so
every log line emitted underneath carries it without being threaded through call
signatures.
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import structlog

from rag_bench.core.settings import AppEnv

_configured = False

# Third-party libraries that log every HTTP request or model file at INFO, which buries
# the pipeline's own events during an index build.
_NOISY_LIBRARIES = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "sentence_transformers",
    "transformers",
)


def configure_logging(app_env: AppEnv = "dev", log_level: str = "INFO") -> None:
    """Configure structlog and the standard library root logger.

    Safe to call more than once; only the first call takes effect.

    Args:
        app_env: ``"prod"`` selects JSON output, ``"dev"`` selects the console renderer.
        log_level: Standard logging level name.
    """
    global _configured
    if _configured:
        return

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=log_level.upper())
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)

    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if app_env == "prod"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to a module name.

    Args:
        name: Usually ``__name__``.

    Returns:
        A structlog logger.
    """
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger


@contextmanager
def log_context(**values: object) -> Iterator[None]:
    """Bind values onto every log line emitted inside the block.

    Args:
        **values: Fields to bind, such as ``run_id`` or ``request_id``.

    Yields:
        Nothing; the binding is ambient.
    """
    tokens = structlog.contextvars.bind_contextvars(**values)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
