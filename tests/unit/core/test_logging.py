import pytest
import structlog

from rag_bench.core import logging as logging_module
from rag_bench.core.logging import configure_logging, get_logger, log_context


@pytest.fixture(autouse=True)
def _reset_logging_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging_module, "_configured", False)
    structlog.contextvars.clear_contextvars()


@pytest.mark.parametrize(
    ("app_env", "renderer"),
    [("prod", structlog.processors.JSONRenderer), ("dev", structlog.dev.ConsoleRenderer)],
)
def test_renderer_is_chosen_by_environment(app_env: str, renderer: type) -> None:
    configure_logging(app_env=app_env)  # type: ignore[arg-type]

    assert isinstance(structlog.get_config()["processors"][-1], renderer)


def test_configuration_is_applied_only_once() -> None:
    configure_logging(app_env="prod")
    configure_logging(app_env="dev")

    assert isinstance(structlog.get_config()["processors"][-1], structlog.processors.JSONRenderer)


def test_log_context_binds_and_unbinds() -> None:
    configure_logging()

    with log_context(run_id="r-1"):
        assert structlog.contextvars.get_contextvars()["run_id"] == "r-1"

    assert "run_id" not in structlog.contextvars.get_contextvars()


def test_log_context_unbinds_even_when_the_block_raises() -> None:
    configure_logging()

    with pytest.raises(RuntimeError), log_context(run_id="r-2"):
        raise RuntimeError("boom")

    assert "run_id" not in structlog.contextvars.get_contextvars()


def test_get_logger_emits_bound_context(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging(app_env="prod")

    with caplog.at_level("WARNING"), log_context(request_id="req-9"):
        get_logger(__name__).warning("thing.happened", detail=1)

    assert '"request_id": "req-9"' in caplog.text
    assert '"event": "thing.happened"' in caplog.text
