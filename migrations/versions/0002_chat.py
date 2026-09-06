"""chat threads and messages for the agent console

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-06

Deliberately additive: this touches none of the four existing tables, so a
downgrade is a clean drop of two tables and nothing else has to be reasoned
about. Note there is no append-only trigger here of the kind classifications
carries -- chat rows legitimately transition running -> done.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_threads",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("owner", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False, server_default="New chat"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # Serves the sidebar: this owner's threads, newest activity first.
    op.create_index(
        "ix_chat_threads_owner_updated",
        "chat_threads",
        ["owner", sa.text("updated_at DESC")],
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("thread_id", sa.BigInteger(), nullable=False),
        sa.Column("turn_id", sa.BigInteger(), nullable=True),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("thought", sa.Text(), nullable=True),
        sa.Column("tool_name", sa.Text(), nullable=True),
        sa.Column("tool_args", postgresql.JSONB(), nullable=True),
        sa.Column("tool_result", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="done"),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("raw_step", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["thread_id"], ["chat_threads.id"], ondelete="CASCADE"),
        # Self-referential: deleting the user message that opened a turn takes
        # that turn's tool calls and its answer with it.
        sa.ForeignKeyConstraint(["turn_id"], ["chat_messages.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("thread_id", "seq", name="uq_chat_messages_seq"),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'tool')", name="ck_chat_messages_role"
        ),
        sa.CheckConstraint(
            "status IN ('done', 'running', 'awaiting_confirm', 'declined', 'error')",
            name="ck_chat_messages_status",
        ),
    )
    op.create_index("ix_chat_messages_thread_seq", "chat_messages", ["thread_id", "seq"])
    op.create_index("ix_chat_messages_turn", "chat_messages", ["turn_id"])
    # Partial, like ix_jobs_pending_run_after: the restart sweep is the only
    # query that looks across every thread for in-flight rows.
    op.create_index(
        "ix_chat_messages_live",
        "chat_messages",
        ["status"],
        postgresql_where=sa.text("status IN ('running', 'awaiting_confirm')"),
    )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chat_threads")
