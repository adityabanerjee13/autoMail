"""Chat persistence against a real Postgres.

Same shape as test_queue.py, and skipped the same way: the behaviour under test
is cascade deletes, a partial unique constraint and JSONB round-tripping, none
of which a fake reproduces honestly.

    docker compose up -d postgres
    TEST_DATABASE_URL=postgresql+psycopg://triage:triage@127.0.0.1:5432/triage_test \
        pytest tests/test_chat_store.py
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from triage.db import chat
from triage.db.models import Base

TEST_DSN = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TEST_DSN, reason="set TEST_DATABASE_URL to run chat store tests"),
]

OWNER = "someone@example.com"


@pytest_asyncio.fixture
async def sessions():
    engine = create_async_engine(TEST_DSN, poolclass=None)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_threads_list_newest_activity_first(sessions):
    async with sessions() as s:
        first = await chat.create_thread(s, OWNER)
        second = await chat.create_thread(s, OWNER)
        await s.commit()
    async with sessions() as s:
        await chat.touch_thread(s, first, title="the older one, touched later")
        await s.commit()
    async with sessions() as s:
        rows = await chat.list_threads(s, OWNER)
    assert [r["id"] for r in rows] == [first, second]
    assert rows[0]["title"] == "the older one, touched later"


async def test_threads_are_scoped_to_their_owner(sessions):
    async with sessions() as s:
        mine = await chat.create_thread(s, OWNER)
        await chat.create_thread(s, "someone-else@example.com")
        await s.commit()
    async with sessions() as s:
        assert [r["id"] for r in await chat.list_threads(s, OWNER)] == [mine]
        # Scoping, not security: get_thread refuses a mismatch, but nothing in
        # this app authenticates the owner in the first place.
        assert await chat.get_thread(s, mine, "someone-else@example.com") is None


async def test_history_for_model_excludes_tool_rows(sessions):
    """The single biggest saving in the token budget.

    A thread where the user ran the queue three times would otherwise replay as
    six tool calls plus six result tables. The transcript still renders all of
    it; only the model's view is thinned.
    """
    async with sessions() as s:
        tid = await chat.create_thread(s, OWNER)
        turn = await chat.add_message(s, tid, role="user", content="what is queued?")
        await chat.add_message(
            s, tid, role="tool", turn_id=turn, tool_name="queue_status", content="pending 3"
        )
        await chat.add_message(s, tid, role="assistant", turn_id=turn, content="Three.")
        await s.commit()
    async with sessions() as s:
        turns = await chat.history_for_model(s, tid)
    assert turns == [
        [
            {"role": "user", "content": "what is queued?"},
            {"role": "assistant", "content": "Three."},
        ]
    ]


async def test_history_for_model_excludes_failed_turns(sessions):
    """A turn that failed is not something to teach the model to imitate."""
    async with sessions() as s:
        tid = await chat.create_thread(s, OWNER)
        turn = await chat.add_message(s, tid, role="user", content="do the thing")
        await chat.add_message(
            s, tid, role="assistant", turn_id=turn, content="model unreachable", status="error"
        )
        await s.commit()
    async with sessions() as s:
        turns = await chat.history_for_model(s, tid)
    assert turns == [[{"role": "user", "content": "do the thing"}]]


async def test_deleting_a_thread_takes_its_messages(sessions):
    async with sessions() as s:
        tid = await chat.create_thread(s, OWNER)
        await chat.add_message(s, tid, role="user", content="hello")
        await s.commit()
    async with sessions() as s:
        assert await chat.delete_thread(s, tid, OWNER) is True
        await s.commit()
    async with sessions() as s:
        assert await chat.messages_for_render(s, tid) == []


async def test_deleting_a_turn_takes_its_tool_calls_and_answer(sessions):
    """The self-referential cascade. What Retry depends on."""
    async with sessions() as s:
        tid = await chat.create_thread(s, OWNER)
        turn = await chat.add_message(s, tid, role="user", content="q")
        await chat.add_message(s, tid, role="tool", turn_id=turn, tool_name="queue_status")
        await chat.add_message(s, tid, role="assistant", turn_id=turn, content="a")
        await s.commit()
    async with sessions() as s:
        removed = await chat.delete_turn_tail(s, turn)
        await s.commit()
    assert removed == 2
    async with sessions() as s:
        rows = await chat.messages_for_render(s, tid)
    # The question survives; only the attempt at answering it is gone.
    assert [r.role for r in rows] == ["user"]


async def test_seq_is_unique_within_a_thread(sessions):
    """The constraint that turns a future interleave into an error, not a mess."""
    from triage.db.models import ChatMessageRow

    async with sessions() as s:
        tid = await chat.create_thread(s, OWNER)
        await chat.add_message(s, tid, role="user", content="one")
        await s.commit()
    with pytest.raises(IntegrityError):
        async with sessions() as s:
            s.add(ChatMessageRow(thread_id=tid, seq=1, role="user", content="clash"))
            await s.commit()


async def test_tool_rows_round_trip_their_jsonb(sessions):
    async with sessions() as s:
        tid = await chat.create_thread(s, OWNER)
        rid = await chat.add_message(
            s,
            tid,
            role="tool",
            tool_name="find_messages",
            tool_args={"query": "bank", "limit": 5},
            status="running",
        )
        await s.commit()
    async with sessions() as s:
        await chat.finish_message(
            s, rid, status="done", tool_result={"ok": True, "text": "2 rows", "meta": {"count": 2}}
        )
        await s.commit()
    async with sessions() as s:
        row = await chat.get_message(s, rid)
    assert row.status == "done"
    assert row.tool_args == {"query": "bank", "limit": 5}
    assert row.tool_result["meta"]["count"] == 2


async def test_pending_confirmation_finds_the_blocked_row(sessions):
    async with sessions() as s:
        tid = await chat.create_thread(s, OWNER)
        await chat.add_message(s, tid, role="user", content="clear it")
        rid = await chat.add_message(
            s, tid, role="tool", tool_name="clear_queue", status="awaiting_confirm"
        )
        await s.commit()
    async with sessions() as s:
        assert await chat.pending_confirmation(s, tid) == rid


async def test_the_restart_sweep_clears_in_flight_rows(sessions):
    """Futures and tasks do not survive a restart; these rows would hang forever."""
    from triage.agent.tools import sweep_interrupted

    async with sessions() as s:
        tid = await chat.create_thread(s, OWNER)
        await chat.add_message(s, tid, role="tool", tool_name="sync_mail", status="running")
        await chat.add_message(s, tid, role="tool", tool_name="clear_queue",
                               status="awaiting_confirm")
        await chat.add_message(s, tid, role="assistant", content="done one", status="done")
        await s.commit()
    async with sessions() as s:
        assert await sweep_interrupted(s) == 2
        await s.commit()
    async with sessions() as s:
        rows = await chat.messages_for_render(s, tid)
    assert [r.status for r in rows] == ["error", "error", "done"]
    assert rows[0].content == "interrupted by a restart"
