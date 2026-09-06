"""Queue tests. These need a live PostgreSQL -- SKIP LOCKED, LISTEN/NOTIFY and
transactional pg_notify have no meaningful fake.

    docker compose up -d postgres
    TEST_DATABASE_URL=postgresql+psycopg://triage:triage@127.0.0.1:5432/triage_test \\
      pytest tests/test_queue.py

The schema is created from the ORM metadata rather than by running Alembic, so
a broken migration shows up in a migration test, not here.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from triage.db import queue, repo
from triage.db.models import Base
from triage.schemas import MessageIn

TEST_DSN = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TEST_DSN, reason="set TEST_DATABASE_URL to run queue tests"),
]


@pytest_asyncio.fixture
async def sessions():
    engine = create_async_engine(TEST_DSN, poolclass=None)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield maker
    finally:
        await engine.dispose()


def _message(gmail_id: str) -> MessageIn:
    return MessageIn(
        gmail_id=gmail_id,
        thread_id="t1",
        from_addr="sender@example.com",
        subject="hello",
        body_clean="body",
        internal_date=datetime.now(UTC),
    )


async def _seed(sessions, gmail_id: str = "g1") -> int:
    async with sessions() as session:
        message, job_id = await repo.ingest_message(session, _message(gmail_id))
        await session.commit()
    assert message is not None
    return job_id


# ---------------------------------------------------------------------------


async def test_message_and_job_commit_together(sessions):
    """The reason there is no Redis: a job cannot exist without its message."""
    job_id = await _seed(sessions)
    async with sessions() as session:
        job = await queue.claim_one(session)
        await session.commit()
    assert job is not None and job.id == job_id
    assert job.status == "running"


async def test_rollback_leaves_neither_row(sessions):
    async with sessions() as session:
        message, _ = await repo.ingest_message(session, _message("g-rollback"))
        assert message is not None
        await session.rollback()

    async with sessions() as session:
        assert await repo.get_message_by_gmail_id(session, "g-rollback") is None
        assert await queue.claim_one(session) is None


async def test_reingesting_the_same_message_is_a_no_op(sessions):
    """Re-running the sync after a stale watermark must not duplicate."""
    await _seed(sessions, "g-dup")
    async with sessions() as session:
        message, job_id = await repo.ingest_message(session, _message("g-dup"))
        await session.commit()
    assert message is None and job_id is None

    async with sessions() as session:
        count = await session.scalar(text("SELECT count(*) FROM jobs"))
    assert count == 1


async def test_skip_locked_hands_each_job_to_one_worker(sessions):
    for i in range(4):
        await _seed(sessions, f"g-skip-{i}")

    claimed: list[int] = []

    async def worker():
        async with sessions() as session:
            job = await queue.claim_one(session)
            if job:
                claimed.append(job.id)
            await session.commit()

    await asyncio.gather(*(worker() for _ in range(4)))
    assert len(claimed) == 4
    assert len(set(claimed)) == 4  # no job claimed twice


async def test_claim_ignores_future_run_after(sessions):
    job_id = await _seed(sessions, "g-future")
    async with sessions() as session:
        await session.execute(
            text("UPDATE jobs SET run_after = now() + interval '1 hour' WHERE id = :id"),
            {"id": job_id},
        )
        await session.commit()
    async with sessions() as session:
        assert await queue.claim_one(session) is None


async def test_failure_backs_off_then_dead_letters(sessions):
    job_id = await _seed(sessions, "g-fail")
    async with sessions() as session:
        status = await queue.fail(session, job_id, "boom", max_attempts=3)
        await session.commit()
    assert status == "pending"

    async with sessions() as session:
        row = (
            await session.execute(
                text("SELECT attempts, run_after, last_error FROM jobs WHERE id = :id"),
                {"id": job_id},
            )
        ).one()
    assert row.attempts == 1
    assert row.run_after > datetime.now(UTC) + timedelta(seconds=20)
    assert "boom" in row.last_error

    for _ in range(2):
        async with sessions() as session:
            status = await queue.fail(session, job_id, "boom", max_attempts=3)
            await session.commit()
    assert status == "dead"


async def test_retry_dead_returns_jobs_to_the_queue(sessions):
    job_id = await _seed(sessions, "g-dead")
    async with sessions() as session:
        await queue.fail(session, job_id, "boom", max_attempts=1)
        await session.commit()

    async with sessions() as session:
        n = await queue.retry_dead(session)
        await session.commit()
    assert n == 1

    async with sessions() as session:
        job = await queue.claim_one(session)
        await session.commit()
    assert job is not None and job.attempts == 0


async def test_stuck_running_jobs_are_requeued(sessions):
    """A worker killed mid-flight leaves 'running' forever otherwise."""
    await _seed(sessions, "g-stuck")
    async with sessions() as session:
        job = await queue.claim_one(session)
        await session.execute(
            text("UPDATE jobs SET started_at = now() - interval '2 hours' WHERE id = :id"),
            {"id": job.id},
        )
        await session.commit()

    async with sessions() as session:
        n = await queue.requeue_stuck(session, timeout_s=600)
        await session.commit()
    assert n == 1

    async with sessions() as session:
        assert await queue.claim_one(session) is not None


async def test_notify_fires_on_commit_and_not_on_rollback(sessions):
    """Workers block on LISTEN; a notification for an uncommitted job would
    wake them to find nothing, and a missing one would leave work sitting."""
    import psycopg

    from triage.db.engine import raw_dsn

    conn = await psycopg.AsyncConnection.connect(raw_dsn(TEST_DSN), autocommit=True)
    try:
        await conn.execute(f"LISTEN {queue.CHANNEL}")
        notifications = conn.notifies()

        async with sessions() as session:
            await repo.ingest_message(session, _message("g-rollback-notify"))
            await session.rollback()

        async with sessions() as session:
            await repo.ingest_message(session, _message("g-notify"))
            await session.commit()

        note = await asyncio.wait_for(anext(notifications), timeout=5)
        assert note.channel == queue.CHANNEL

        # The rolled-back insert must not have produced a second notification.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(notifications), timeout=1)
    finally:
        await conn.close()


async def test_counts_group_by_status(sessions):
    await _seed(sessions, "g-count-1")
    job_id = await _seed(sessions, "g-count-2")
    async with sessions() as session:
        await queue.complete(session, job_id)
        counts = await queue.counts(session)
        await session.commit()
    assert counts["pending"] == 1
    assert counts["done"] == 1
