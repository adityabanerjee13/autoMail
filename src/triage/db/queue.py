"""Postgres-as-queue: SELECT ... FOR UPDATE SKIP LOCKED plus LISTEN/NOTIFY.

Two rules hold this together:

* ``enqueue`` never opens its own transaction. It takes the caller's session so
  the ``messages`` INSERT and the ``jobs`` INSERT commit together. A job that
  can reference a message that does not exist is the failure mode this design
  exists to prevent.
* Workers do not poll. They hold a dedicated connection on LISTEN, drain the
  queue when woken, then block again.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import psycopg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from triage.config import settings
from triage.db.engine import raw_dsn
from triage.schemas import Job

CHANNEL = "triage_jobs"

#: Attempt N waits BACKOFF_BASE_S * 2**(N-1), capped. 30s, 60s, 120s, 240s...
BACKOFF_BASE_S = 30
BACKOFF_CAP_S = 3600


def _backoff(attempts: int) -> timedelta:
    return timedelta(seconds=min(BACKOFF_BASE_S * (2 ** max(attempts - 1, 0)), BACKOFF_CAP_S))


async def enqueue(session: AsyncSession, message_id: int, *, notify: bool = True) -> int:
    """Insert a pending job in the caller's transaction and wake a worker.

    pg_notify inside the transaction is itself transactional: the notification
    is delivered on commit and discarded on rollback, so a worker is never
    woken for a job that does not exist.
    """
    row = await session.execute(
        text(
            """
            INSERT INTO jobs (message_id, status, attempts, run_after)
            VALUES (:message_id, 'pending', 0, now())
            RETURNING id
            """
        ),
        {"message_id": message_id},
    )
    job_id = row.scalar_one()
    if notify:
        await session.execute(
            text("SELECT pg_notify(:channel, :payload)"),
            {"channel": CHANNEL, "payload": str(job_id)},
        )
    return job_id


async def claim_one(session: AsyncSession) -> Job | None:
    """Atomically claim the oldest runnable job. Returns None when idle."""
    result = await session.execute(
        text(
            """
            UPDATE jobs SET status = 'running', started_at = now()
            WHERE id = (
              SELECT id FROM jobs
              WHERE status = 'pending' AND run_after <= now()
              ORDER BY id
              LIMIT 1
              FOR UPDATE SKIP LOCKED
            )
            RETURNING id, message_id, status, attempts, run_after,
                      last_error, started_at, created_at
            """
        )
    )
    row = result.mappings().first()
    return Job.model_validate(dict(row)) if row else None


async def pending_job_for(session: AsyncSession, message_id: int) -> int | None:
    """The queued job for this message, if it already has one.

    Without this, asking to process a message that is already in the queue
    enqueues a second job for it. The duplicate is harmless -- the runner's
    idempotency guard skips it -- but it inflates the queue depth the console
    reports, which makes the number untrustworthy exactly when someone is
    watching it.
    """
    return await session.scalar(
        text(
            """
            SELECT id FROM jobs
             WHERE message_id = :message_id AND status = 'pending'
             ORDER BY id LIMIT 1
            """
        ),
        {"message_id": message_id},
    )


async def remove_pending_for(session: AsyncSession, message_id: int) -> int:
    """Take one message back out of the queue.

    Only 'pending' rows, for the same reason ``clear_pending_jobs`` skips
    'running' ones: deleting a claimed job orphans the runner holding it, which
    finishes the pipeline and then updates a row that no longer exists. The
    message itself is untouched, so it stays in the unprocessed list and can be
    queued again.
    """
    result = await session.execute(
        text(
            """
            DELETE FROM jobs
             WHERE message_id = :message_id AND status = 'pending'
            RETURNING id
            """
        ),
        {"message_id": message_id},
    )
    return len(result.all())


async def claim_job(session: AsyncSession, job_id: int) -> Job | None:
    """Claim one named job, or None if it is gone or already taken.

    The console's per-message "Process" button. ``claim_one`` deliberately
    takes the oldest pending job, which is right for a worker draining a queue
    and wrong for "run this one" -- it would claim somebody else's message and
    look like the button processed the wrong mail.

    Same SKIP LOCKED discipline as ``claim_one``, so racing a worker for this
    row loses cleanly instead of double-processing it.
    """
    result = await session.execute(
        text(
            """
            UPDATE jobs SET status = 'running', started_at = now()
            WHERE id = (
              SELECT id FROM jobs
              WHERE id = :job_id AND status = 'pending'
              FOR UPDATE SKIP LOCKED
            )
            RETURNING id, message_id, status, attempts, run_after,
                      last_error, started_at, created_at
            """
        ),
        {"job_id": job_id},
    )
    row = result.mappings().first()
    return Job.model_validate(dict(row)) if row else None


async def complete(session: AsyncSession, job_id: int) -> None:
    await session.execute(
        text("UPDATE jobs SET status = 'done', last_error = NULL WHERE id = :id"),
        {"id": job_id},
    )


async def fail(
    session: AsyncSession,
    job_id: int,
    error: str,
    *,
    max_attempts: int | None = None,
) -> str:
    """Record a failure, then either schedule a retry or dead-letter the job.

    Returns the resulting status so the caller can log it without a re-read.
    """
    limit = max_attempts if max_attempts is not None else settings.job_max_attempts
    result = await session.execute(
        text("SELECT attempts FROM jobs WHERE id = :id FOR UPDATE"),
        {"id": job_id},
    )
    attempts = (result.scalar_one_or_none() or 0) + 1
    dead = attempts >= limit
    await session.execute(
        text(
            """
            UPDATE jobs
               SET status = :status,
                   attempts = :attempts,
                   last_error = :error,
                   run_after = :run_after,
                   started_at = NULL
             WHERE id = :id
            """
        ),
        {
            "id": job_id,
            "status": "dead" if dead else "pending",
            "attempts": attempts,
            # Truncated: a full traceback in every row bloats the table and the
            # first lines are what you actually read.
            "error": error[:4000],
            "run_after": datetime.now(UTC) + _backoff(attempts),
        },
    )
    return "dead" if dead else "pending"


async def requeue_stuck(session: AsyncSession, timeout_s: int | None = None) -> int:
    """Return jobs whose worker died mid-flight to the pending pool.

    A worker killed between claim and completion leaves a row in 'running'
    forever; nothing else in the system will ever pick it up.
    """
    timeout = timeout_s if timeout_s is not None else settings.job_stuck_timeout_s
    result = await session.execute(
        text(
            """
            UPDATE jobs
               SET status = 'pending',
                   started_at = NULL,
                   last_error = 'requeued: exceeded running timeout'
             WHERE status = 'running'
               AND started_at < now() - make_interval(secs => :timeout)
            RETURNING id
            """
        ),
        {"timeout": timeout},
    )
    ids = [r[0] for r in result]
    if ids:
        await session.execute(
            text("SELECT pg_notify(:channel, 'requeue')"), {"channel": CHANNEL}
        )
    return len(ids)


async def retry_dead(session: AsyncSession, limit: int = 100) -> int:
    """Give dead-lettered jobs another pass (hourly, from the scheduler).

    Attempts are reset because the usual cause of a dead letter here is vLLM
    having been down, not the message being unprocessable.
    """
    result = await session.execute(
        text(
            """
            UPDATE jobs
               SET status = 'pending', attempts = 0, run_after = now()
             WHERE id IN (
               SELECT id FROM jobs WHERE status = 'dead' ORDER BY id LIMIT :limit
             )
            RETURNING id
            """
        ),
        {"limit": limit},
    )
    ids = [r[0] for r in result]
    if ids:
        await session.execute(
            text("SELECT pg_notify(:channel, 'retry_dead')"), {"channel": CHANNEL}
        )
    return len(ids)


async def counts(session: AsyncSession) -> dict[str, int]:
    result = await session.execute(text("SELECT status, count(*) FROM jobs GROUP BY status"))
    return {status: n for status, n in result}


async def listen(stop: asyncio.Event | None = None) -> AsyncIterator[str]:
    """Yield once per NOTIFY on the triage_jobs channel.

    Deliberately a bare psycopg connection rather than one from the SQLAlchemy
    pool: this connection is held for the process lifetime and would otherwise
    never return to the pool.

    The yielded payload carries no meaning beyond "something changed" -- the
    worker always drains via claim_one rather than trusting the payload, which
    is what makes a missed or coalesced notification harmless.
    """
    conn = await psycopg.AsyncConnection.connect(raw_dsn(), autocommit=True)
    try:
        await conn.execute(f"LISTEN {CHANNEL}")
        gen = conn.notifies()
        async for note in gen:
            yield note.payload
            if stop is not None and stop.is_set():
                await gen.aclose()
                break
    finally:
        await conn.close()
