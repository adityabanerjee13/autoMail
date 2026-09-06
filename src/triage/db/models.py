"""SQLAlchemy table definitions.

These are the Alembic autogenerate target and the query surface for
``db/repo.py``. Nothing above ``db/`` should import from this module.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from triage.config import settings

#: Must match the vector(N) width in the migration. Changing the embedding
#: model to one of a different width is a migration, not a config change.
EMBEDDING_DIM = settings.embedding_dim


class Base(DeclarativeBase):
    pass


class MessageRow(Base):
    """Immutable record of what arrived. Only fingerprint and embedding are
    ever written after insert, and only by the pipeline's post-actions."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    gmail_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    message_id_hdr: Mapped[str | None] = mapped_column(Text)
    from_addr: Mapped[str] = mapped_column(Text, nullable=False)
    to_addrs: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    cc_addrs: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    subject: Mapped[str | None] = mapped_column(Text)
    body_clean: Mapped[str | None] = mapped_column(Text)
    snippet: Mapped[str | None] = mapped_column(Text)
    labels: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    headers: Mapped[dict | None] = mapped_column(JSONB)
    fingerprint: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    internal_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_messages_thread_id", "thread_id"),
        Index("ix_messages_from_addr", "from_addr"),
        Index("ix_messages_fingerprint", "fingerprint"),
        Index("ix_messages_internal_date", "internal_date"),
    )


class ClassificationRow(Base):
    """Append-only. There is no code path in this repository that UPDATEs it.

    The three version columns are load-bearing: without them the accumulated
    dataset is a mixture of incompatible labellers and is worthless for the
    phase-2 supervised classifier.
    """

    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    is_important: Mapped[bool] = mapped_column(Boolean, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Kept even though payload holds the parsed object: schema-validation
    # failures and truncated generations cannot be reconstructed from parsed
    # output.
    raw_response: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_classifications_message_id", "message_id"),
        # Serves the idempotency guard in runner step 2.
        Index(
            "ix_classifications_idempotency",
            "message_id",
            "model_id",
            "prompt_version",
        ),
        Index("ix_classifications_created_at", "created_at"),
    )


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    run_after: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # The claim query's covering index: pending jobs ordered by run_after.
        Index(
            "ix_jobs_pending_run_after",
            "run_after",
            postgresql_where=text("status = 'pending'"),
        ),
        # Serves the stuck-job sweep.
        Index(
            "ix_jobs_running_started_at",
            "started_at",
            postgresql_where=text("status = 'running'"),
        ),
        Index("ix_jobs_message_id", "message_id"),
    )


class SyncStateRow(Base):
    """Single row, id = 1. Enforced by the CHECK constraint in the migration."""

    __tablename__ = "sync_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    history_id: Mapped[str | None] = mapped_column(Text)
    watch_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_reconcile_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChatThreadRow(Base):
    """One conversation with the agent.

    ``owner`` is the connected mailbox address -- see ``db/chat.owner_key``. It
    is scoping, not security: this app has no authentication and binds to
    loopback. It exists so that reconnecting a different Gmail account does not
    mix two people's conversations together on a shared box.
    """

    __tablename__ = "chat_threads"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="New chat")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_chat_threads_owner_updated", "owner", "updated_at"),)


class ChatMessageRow(Base):
    """One row per user message, per tool call, and per assistant reply.

    A row per tool call rather than one blob per turn, for four reasons. The
    turn is written incrementally, so the page's two-second poll shows real
    progress instead of a spinner for a minute; ``awaiting_confirm`` needs a
    durable, addressable row that a confirmation POST can target and a restart
    can sweep; rendering stays an ordered scan like every other template here;
    and replaying a thread to the model becomes a WHERE on ``role``.

    Unlike ``classifications`` this table is *not* append-only -- rows
    legitimately move ``running -> done`` -- so it carries no immutability
    trigger.
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    thread_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("chat_threads.id", ondelete="CASCADE"), nullable=False
    )
    #: The role='user' row that started this turn. Self-referential, so deleting
    #: a user message takes its tool calls and its answer with it.
    turn_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("chat_messages.id", ondelete="CASCADE")
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    thought: Mapped[str | None] = mapped_column(Text)
    tool_name: Mapped[str | None] = mapped_column(Text)
    tool_args: Mapped[dict | None] = mapped_column(JSONB)
    #: {"ok": bool, "text": str, "summary": str, "meta": {...}}. ``text`` is the
    #: model-facing rendering, kept for the same reason classifications keeps
    #: raw_response: when the agent behaves oddly, what it was *told* is the
    #: first thing you need and cannot reconstruct.
    tool_result: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="done")
    error_code: Mapped[str | None] = mapped_column(Text)
    #: Exactly what the model emitted, before parsing.
    raw_step: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("thread_id", "seq", name="uq_chat_messages_seq"),
        CheckConstraint(
            "role IN ('user', 'assistant', 'tool')", name="ck_chat_messages_role"
        ),
        CheckConstraint(
            "status IN ('done', 'running', 'awaiting_confirm', 'declined', 'error')",
            name="ck_chat_messages_status",
        ),
        Index("ix_chat_messages_thread_seq", "thread_id", "seq"),
        Index("ix_chat_messages_turn", "turn_id"),
        # Serves the restart sweep, which is the only query that looks for
        # in-flight rows across every thread.
        Index(
            "ix_chat_messages_live",
            "status",
            postgresql_where=text("status IN ('running', 'awaiting_confirm')"),
        ),
    )
