"""initial schema: messages, classifications, jobs, sync_state

Revision ID: 0001
Revises:
Create Date: 2026-09-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

#: Must match EMBEDDING_DIM / the width of EMBEDDING_MODEL. bge-m3 is 1024,
#: e5-base is 768. Switching models is a migration, not a config change.
EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # -- messages ----------------------------------------------------------
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("gmail_id", sa.Text, nullable=False, unique=True),
        sa.Column("thread_id", sa.Text, nullable=False),
        sa.Column("message_id_hdr", sa.Text),
        sa.Column("from_addr", sa.Text, nullable=False),
        sa.Column("to_addrs", postgresql.ARRAY(sa.Text)),
        sa.Column("cc_addrs", postgresql.ARRAY(sa.Text)),
        sa.Column("subject", sa.Text),
        sa.Column("body_clean", sa.Text),
        sa.Column("snippet", sa.Text),
        sa.Column("labels", postgresql.ARRAY(sa.Text)),
        sa.Column("headers", postgresql.JSONB),
        sa.Column("fingerprint", sa.Text),
        sa.Column("embedding", Vector(EMBEDDING_DIM)),
        sa.Column("internal_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_messages_thread_id", "messages", ["thread_id"])
    op.create_index("ix_messages_from_addr", "messages", ["from_addr"])
    op.create_index("ix_messages_fingerprint", "messages", ["fingerprint"])
    op.create_index("ix_messages_internal_date", "messages", ["internal_date"])
    # Cosine, because embeddings are stored L2-normalised (see llm/embeddings).
    op.execute(
        "CREATE INDEX ix_messages_embedding_hnsw ON messages "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # -- classifications ---------------------------------------------------
    op.create_table(
        "classifications",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "message_id",
            sa.BigInteger,
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("is_important", sa.Boolean, nullable=False),
        sa.Column("category", sa.Text, nullable=False),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("raw_response", sa.Text, nullable=False),
        sa.Column("model_id", sa.Text, nullable=False),
        sa.Column("prompt_version", sa.Text, nullable=False),
        sa.Column("schema_version", sa.Text, nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("latency_ms", sa.Integer),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "source IN ('llm', 'dedup', 'human')", name="ck_classifications_source"
        ),
    )
    op.create_index("ix_classifications_message_id", "classifications", ["message_id"])
    op.create_index(
        "ix_classifications_idempotency",
        "classifications",
        ["message_id", "model_id", "prompt_version"],
    )
    op.create_index("ix_classifications_created_at", "classifications", ["created_at"])
    # Append-only is a rule the application follows; this makes the database
    # enforce it, so a stray UPDATE in a future migration or a psql session
    # fails loudly instead of silently destroying the labelling history.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION classifications_no_update() RETURNS trigger AS $$
        BEGIN
          RAISE EXCEPTION 'classifications is append-only; insert a new row instead';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "CREATE TRIGGER trg_classifications_no_update BEFORE UPDATE ON classifications "
        "FOR EACH ROW EXECUTE FUNCTION classifications_no_update()"
    )

    # -- jobs --------------------------------------------------------------
    op.create_table(
        "jobs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "message_id",
            sa.BigInteger,
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "run_after", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("last_error", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'dead')", name="ck_jobs_status"
        ),
    )
    # Partial index: the claim query only ever looks at pending rows, and this
    # keeps the index from growing with every completed job.
    op.create_index(
        "ix_jobs_pending_run_after",
        "jobs",
        ["run_after"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "ix_jobs_running_started_at",
        "jobs",
        ["started_at"],
        postgresql_where=sa.text("status = 'running'"),
    )
    op.create_index("ix_jobs_message_id", "jobs", ["message_id"])

    # -- sync_state --------------------------------------------------------
    op.create_table(
        "sync_state",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("history_id", sa.Text),
        sa.Column("watch_expiry", sa.DateTime(timezone=True)),
        sa.Column("last_reconcile_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("id = 1", name="ck_sync_state_singleton"),
    )
    op.execute("INSERT INTO sync_state (id) VALUES (1) ON CONFLICT DO NOTHING")


def downgrade() -> None:
    op.drop_table("sync_state")
    op.drop_table("jobs")
    op.execute("DROP TRIGGER IF EXISTS trg_classifications_no_update ON classifications")
    op.execute("DROP FUNCTION IF EXISTS classifications_no_update()")
    op.drop_table("classifications")
    op.drop_table("messages")
    # The vector extension is left in place: other databases in the cluster may
    # be using it, and dropping it would cascade.
