"""Liveness and readiness.

The two answer different questions. Liveness asks whether the process should be
restarted; readiness asks whether it should receive traffic. Conflating them means a
database blip restarts a perfectly healthy process.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Response, status

from rag_bench.api.dependencies import EngineDep, SettingsDep
from rag_bench.api.schemas import HealthResponse
from rag_bench.core.exceptions import RagBenchError
from rag_bench.core.llm import check_llm_health
from rag_bench.db.session import check_database

router = APIRouter(prefix="/health", tags=["health"])

_OK = "ok"


@router.get("/live", response_model=HealthResponse, summary="Is the process alive")
def live() -> HealthResponse:
    """Report that the process is running.

    Deliberately checks nothing external, so a dependency outage never causes an
    orchestrator to kill a process that is working fine.
    """
    return HealthResponse(status=_OK)


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Can the service serve traffic",
    responses={503: {"model": HealthResponse, "description": "A dependency is unavailable."}},
)
def ready(settings: SettingsDep, engine: EngineDep, response: Response) -> HealthResponse:
    """Check every dependency a request would need.

    Returns 503 when any of them is down, so a load balancer stops sending traffic
    rather than letting requests fail one by one.
    """
    checks = {
        "database": _probe(lambda: check_database(engine)),
        "llm": _probe(lambda: check_llm_health(settings.llm)),
    }
    healthy = all(result == _OK for result in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status=_OK if healthy else "degraded", checks=checks)


def _probe(check: Callable[[], None]) -> str:
    """Run one dependency check and reduce it to a short status string."""
    try:
        check()
    except RagBenchError as exc:
        return f"unavailable: {exc.code}"
    return _OK
