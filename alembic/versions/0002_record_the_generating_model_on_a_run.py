"""record the generating model on a run

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-27 13:32:49.501381
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the migration."""
    # A server default is required because the table may already hold rows, and those
    # runs genuinely have no recorded model. It is dropped afterwards so new inserts
    # must state one rather than silently inheriting a blank.
    op.add_column(
        "benchmark_runs",
        sa.Column("llm_provider", sa.String(length=64), nullable=False, server_default=""),
    )
    op.add_column(
        "benchmark_runs",
        sa.Column("llm_model", sa.String(length=200), nullable=False, server_default=""),
    )
    with op.batch_alter_table("benchmark_runs") as batch:
        batch.alter_column("llm_provider", server_default=None)
        batch.alter_column("llm_model", server_default=None)


def downgrade() -> None:
    """Revert the migration."""
    op.drop_column("benchmark_runs", "llm_model")
    op.drop_column("benchmark_runs", "llm_provider")
