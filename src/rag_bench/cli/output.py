"""The CLI's user-facing output layer.

This is the one place in the package allowed to write to stdout for a human. Everything
else logs through structlog.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import typer
from rich.console import Console
from rich.table import Table

from rag_bench.core.exceptions import RagBenchError
from rag_bench.core.logging import configure_logging
from rag_bench.core.settings import get_settings

console = Console()
error_console = Console(stderr=True)


@contextmanager
def reporting_errors() -> Iterator[None]:
    """Turn a project error into a clean message and a non-zero exit code.

    A stack trace is the right output for a bug, and the wrong output for a missing
    corpus or a typo in a config file.

    Yields:
        Nothing; the handling is ambient.

    Raises:
        Exit: With code 1 when a project error escapes the block.
    """
    try:
        yield
    except RagBenchError as exc:
        error_console.print(f"[bold red]{exc.code}[/bold red] {exc.message}")
        for key, value in exc.details.items():
            error_console.print(f"  {key}: {value}", style="dim")
        raise typer.Exit(code=1) from exc


def setup_logging() -> None:
    """Configure logging from the environment, once per invocation."""
    settings = get_settings()
    configure_logging(app_env=settings.app_env, log_level=settings.log_level)


def key_value_table(title: str, rows: dict[str, object]) -> None:
    """Print a two-column summary table.

    Args:
        title: Table heading.
        rows: Field names mapped to values.
    """
    table = Table(title=title, show_header=False, title_justify="left")
    table.add_column(style="bold")
    table.add_column()
    for key, value in rows.items():
        table.add_row(key, str(value))
    console.print(table)
