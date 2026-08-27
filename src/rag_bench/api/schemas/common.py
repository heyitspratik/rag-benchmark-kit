"""Envelopes shared by every endpoint.

One error shape and one list shape, so a client writes the handling once rather than
once per resource.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """The body of an error response."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Stable machine-readable code, e.g. UNKNOWN_COMPONENT.")
    message: str = Field(description="Human-readable explanation, safe to display.")
    details: dict[str, object] = Field(
        default_factory=dict, description="Structured context about the failure."
    )
    request_id: str = Field(description="Correlates this response with the server logs.")


class ErrorResponse(BaseModel):
    """The single error shape returned by every failing endpoint."""

    model_config = ConfigDict(frozen=True)

    error: ErrorDetail


class Page[T](BaseModel):
    """One page of a cursor-paginated list.

    Cursor rather than offset paging: benchmark runs are inserted while a client is
    reading, and an offset would silently skip or repeat rows when that happens.
    """

    model_config = ConfigDict(frozen=True)

    items: list[T]
    next_cursor: str | None = Field(
        default=None,
        description="Pass as `cursor` to fetch the next page. Null when there are no more.",
    )
