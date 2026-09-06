"""Historical import over IMAP. Run it overnight, once.

    python -m triage.backfill --before 2026-09-01
    python -m triage.backfill --before 2026-09-01 --no-enqueue   # store only

``--no-enqueue`` is build-order step 1: get ingestion and parsing right, verify
that re-running produces no duplicates, and only then let the LLM near 50,000
messages. Jobs for the stored messages can be created later with
``--enqueue-existing``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

from sqlalchemy import text

from triage.config import Settings, get_settings
from triage.db import queue, repo
from triage.db.engine import configure_event_loop_policy, sessionmaker_for
from triage.ingest.base import to_message_in
from triage.ingest.imap import ImapSource

log = logging.getLogger("triage.backfill")

PROGRESS_EVERY = 250


async def backfill(
    before: datetime,
    *,
    enqueue: bool = True,
    limit: int | None = None,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    sessions = sessionmaker_for()
    source = ImapSource(settings)

    stored = duplicates = failed = 0
    try:
        async for parsed in source.backfill(before):
            try:
                async with sessions() as session:
                    message = await repo.insert_message(session, to_message_in(parsed))
                    if message is not None and enqueue:
                        # Same transaction as the insert, exactly as in the
                        # steady-state path.
                        await queue.enqueue(session, message.id, notify=False)
                    await session.commit()
                if message is None:
                    duplicates += 1
                else:
                    stored += 1
            except Exception:  # noqa: BLE001 - one bad message must not end the run
                failed += 1
                log.exception("failed on gmail_id=%s", parsed.gmail_id)

            total = stored + duplicates + failed
            if total % PROGRESS_EVERY == 0:
                log.info("progress: stored=%d dup=%d failed=%d", stored, duplicates, failed)
            if limit and total >= limit:
                break
    finally:
        await source.close()

    if enqueue and stored:
        # One NOTIFY at the end rather than tens of thousands during the run:
        # workers drain the whole queue on a single wake anyway.
        async with sessions() as session:
            await session.execute(text("SELECT pg_notify('triage_jobs', 'backfill')"))
            await session.commit()

    log.info("backfill complete: stored=%d duplicates=%d failed=%d", stored, duplicates, failed)
    return {"stored": stored, "duplicates": duplicates, "failed": failed}


async def enqueue_existing(limit: int = 100_000) -> int:
    """Create jobs for messages that have no classification and no live job.

    The follow-up to a ``--no-enqueue`` import, and the repair path if jobs are
    ever lost.
    """
    sessions = sessionmaker_for()
    async with sessions() as session:
        result = await session.execute(
            text(
                """
                SELECT m.id FROM messages m
                 WHERE NOT EXISTS (SELECT 1 FROM classifications c WHERE c.message_id = m.id)
                   AND NOT EXISTS (
                        SELECT 1 FROM jobs j
                         WHERE j.message_id = m.id AND j.status IN ('pending', 'running')
                   )
                 ORDER BY m.internal_date DESC
                 LIMIT :limit
                """
            ),
            {"limit": limit},
        )
        ids = [r[0] for r in result]
        for message_id in ids:
            await queue.enqueue(session, message_id, notify=False)
        if ids:
            await session.execute(text("SELECT pg_notify('triage_jobs', 'enqueue_existing')"))
        await session.commit()
    log.info("enqueued %d message(s)", len(ids))
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC),
        default=datetime.now(UTC),
        help="import messages older than this date (YYYY-MM-DD)",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N messages")
    parser.add_argument(
        "--no-enqueue",
        action="store_true",
        help="store messages without queueing them for classification",
    )
    parser.add_argument(
        "--enqueue-existing",
        action="store_true",
        help="skip the import; queue already-stored messages that have no classification",
    )
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        stream=sys.stdout,
    )

    configure_event_loop_policy()

    if args.enqueue_existing:
        asyncio.run(enqueue_existing())
        return
    asyncio.run(
        backfill(args.before, enqueue=not args.no_enqueue, limit=args.limit, settings=settings)
    )


if __name__ == "__main__":
    main()
