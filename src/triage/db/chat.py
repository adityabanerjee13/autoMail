"""Queries for the chat tables.

Separate from ``repo.py``, whose docstring claims every query in the system but
whose subject is mail. Chat is a different bounded context: two tables, no join
to the mail tables, and a different lifecycle. Same house rules apply -- nothing
here commits, and nothing here returns an ORM row above the ``db/`` layer.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from triage.config import Settings, get_settings
from triage.db.models import ChatMessageRow, ChatThreadRow
from triage.ingest import auth

TITLE_LEN = 60


def owner_key(settings: Settings | None = None) -> str:
    """Which mailbox these threads belong to.

    Three steps, not two: ``account_email`` reads an ``"account"`` key out of
    token.json that is not always written, so being OAuth-connected does not
    guarantee an address. A thread must never be orphaned by a missing dict
    key, hence the literal fallback.

    This is **scoping, not security.** The API binds to loopback and has no
    authentication of any kind; anyone who can reach it can read every thread
    regardless of this value.
    """
    settings = settings or get_settings()
    return (
        auth.account_email(settings)
        or (auth.imap_credentials(settings) or (None,))[0]
        or "local"
    )


# -- threads ----------------------------------------------------------------


async def create_thread(session: AsyncSession, owner: str) -> int:
    row = ChatThreadRow(owner=owner)
    session.add(row)
    await session.flush()
    return row.id


async def list_threads(session: AsyncSession, owner: str, *, limit: int = 50) -> list[dict]:
    stmt = (
        select(ChatThreadRow)
        .where(ChatThreadRow.owner == owner)
        # id DESC breaks ties: two threads created in the same second would
        # otherwise reorder themselves between polls.
        .order_by(desc(ChatThreadRow.updated_at), desc(ChatThreadRow.id))
        .limit(limit)
    )
    return [
        {"id": r.id, "title": r.title, "updated_at": r.updated_at}
        for r in (await session.scalars(stmt)).all()
    ]


async def get_thread(session: AsyncSession, thread_id: int, owner: str) -> dict | None:
    row = await session.get(ChatThreadRow, thread_id)
    if row is None or row.owner != owner:
        return None
    return {"id": row.id, "title": row.title, "updated_at": row.updated_at}


async def newest_thread_id(session: AsyncSession, owner: str) -> int | None:
    stmt = (
        select(ChatThreadRow.id)
        .where(ChatThreadRow.owner == owner)
        .order_by(desc(ChatThreadRow.updated_at), desc(ChatThreadRow.id))
        .limit(1)
    )
    return await session.scalar(stmt)


async def delete_thread(session: AsyncSession, thread_id: int, owner: str) -> bool:
    row = await session.get(ChatThreadRow, thread_id)
    if row is None or row.owner != owner:
        return False
    await session.delete(row)
    return True


async def touch_thread(session: AsyncSession, thread_id: int, *, title: str | None = None) -> None:
    values: dict[str, Any] = {"updated_at": func.now()}
    if title is not None:
        values["title"] = title[:TITLE_LEN]
    await session.execute(
        update(ChatThreadRow).where(ChatThreadRow.id == thread_id).values(**values)
    )


# -- messages ---------------------------------------------------------------


async def _next_seq(session: AsyncSession, thread_id: int) -> int:
    """One past the highest seq in this thread.

    Safe without locking only because ``AgentRunner`` allows one turn in flight
    at a time process-wide, so there is never a second writer racing for the
    same number. The UNIQUE constraint is what turns a future mistake here into
    an error rather than a silently interleaved transcript.
    """
    current = await session.scalar(
        select(func.max(ChatMessageRow.seq)).where(ChatMessageRow.thread_id == thread_id)
    )
    return (current or 0) + 1


async def add_message(
    session: AsyncSession,
    thread_id: int,
    *,
    role: str,
    content: str = "",
    turn_id: int | None = None,
    thought: str | None = None,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    tool_result: dict | None = None,
    status: str = "done",
    error_code: str | None = None,
    raw_step: str | None = None,
    latency_ms: int | None = None,
) -> int:
    row = ChatMessageRow(
        thread_id=thread_id,
        turn_id=turn_id,
        seq=await _next_seq(session, thread_id),
        role=role,
        content=content,
        thought=thought,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        status=status,
        error_code=error_code,
        raw_step=raw_step,
        latency_ms=latency_ms,
    )
    session.add(row)
    await session.flush()
    return row.id


async def finish_message(
    session: AsyncSession,
    message_id: int,
    *,
    status: str,
    content: str | None = None,
    tool_result: dict | None = None,
    error_code: str | None = None,
    latency_ms: int | None = None,
) -> None:
    values: dict[str, Any] = {"status": status}
    if content is not None:
        values["content"] = content
    if tool_result is not None:
        values["tool_result"] = tool_result
    if error_code is not None:
        values["error_code"] = error_code
    if latency_ms is not None:
        values["latency_ms"] = latency_ms
    await session.execute(
        update(ChatMessageRow).where(ChatMessageRow.id == message_id).values(**values)
    )


async def messages_for_render(session: AsyncSession, thread_id: int) -> list[ChatMessageRow]:
    """The whole transcript, in order. Everything is shown, including tool rows."""
    stmt = (
        select(ChatMessageRow)
        .where(ChatMessageRow.thread_id == thread_id)
        .order_by(ChatMessageRow.seq)
    )
    return list((await session.scalars(stmt)).all())


async def history_for_model(
    session: AsyncSession, thread_id: int, *, before_turn: int | None = None
) -> list[list[dict[str, str]]]:
    """Prior turns as ``[[{role, content}, ...], ...]``, oldest turn first.

    **Tool rows are excluded**, and that is the single biggest saving in the
    token budget: a thread where the user ran the queue three times replays as
    six short lines rather than six tool calls and six result tables. The
    transcript still renders everything; only the model's view is thinned.

    Rows that are not ``done`` are excluded too -- a failed or half-written turn
    is not something to teach the model to imitate.
    """
    stmt = (
        select(ChatMessageRow)
        .where(
            ChatMessageRow.thread_id == thread_id,
            ChatMessageRow.role.in_(("user", "assistant")),
            ChatMessageRow.status == "done",
            ChatMessageRow.content != "",
        )
        .order_by(ChatMessageRow.seq)
    )
    if before_turn is not None:
        stmt = stmt.where(ChatMessageRow.id < before_turn)

    turns: list[list[dict[str, str]]] = []
    for row in (await session.scalars(stmt)).all():
        if row.role == "user" or not turns:
            turns.append([])
        turns[-1].append({"role": row.role, "content": row.content})
    return [t for t in turns if t]


async def get_message(session: AsyncSession, message_id: int) -> ChatMessageRow | None:
    return await session.get(ChatMessageRow, message_id)


async def pending_confirmation(session: AsyncSession, thread_id: int) -> int | None:
    stmt = (
        select(ChatMessageRow.id)
        .where(
            ChatMessageRow.thread_id == thread_id,
            ChatMessageRow.status == "awaiting_confirm",
        )
        .order_by(ChatMessageRow.seq)
        .limit(1)
    )
    return await session.scalar(stmt)


async def delete_turn_tail(session: AsyncSession, turn_id: int) -> int:
    """Drop everything a turn produced, keeping the user message itself.

    What Retry needs: the question stays, the failed attempt at answering it
    goes.
    """
    result = await session.execute(
        delete(ChatMessageRow).where(ChatMessageRow.turn_id == turn_id)
    )
    return result.rowcount or 0


async def last_turn_id(session: AsyncSession, thread_id: int) -> int | None:
    stmt = (
        select(ChatMessageRow.id)
        .where(ChatMessageRow.thread_id == thread_id, ChatMessageRow.role == "user")
        .order_by(desc(ChatMessageRow.seq))
        .limit(1)
    )
    return await session.scalar(stmt)
