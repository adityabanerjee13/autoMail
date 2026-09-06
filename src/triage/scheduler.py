"""Periodic maintenance. Not a cron loop over the pipeline.

Ingestion is event-driven (Pub/Sub) and the worker is event-driven
(LISTEN/NOTIFY). Everything here is either a safety net for those two or
housekeeping.

| Job                                    | Interval |
| Renew Gmail watch()                    | daily    |
| Reconciliation sweep via history.list  | hourly   |
| Requeue jobs stuck in 'running'        | 5 min    |
| Retry dead-lettered jobs               | hourly   |
| Eval against golden.jsonl              | weekly   |
| Prune old fingerprints                 | weekly   |

The hourly reconciliation is the important one. Pub/Sub is at-least-once but
not guaranteed-delivery; the sweep is the only thing that catches a
notification that never arrived.

Runs inside the triage-sync process.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from triage.config import Settings, get_settings
from triage.db import queue, repo
from triage.db.engine import sessionmaker_for

log = logging.getLogger("triage.scheduler")


async def reconcile(daemon) -> None:
    """Independent watermark sweep. The safety net under Pub/Sub."""
    result = await daemon.sync_once(reconcile=True)
    if result.get("stored"):
        log.warning(
            "reconciliation found %d message(s) Pub/Sub did not deliver", result["stored"]
        )


async def renew_watch(daemon) -> None:
    """watch() expires after 7 days. Daily, so a single failure has margin."""
    try:
        await daemon.ensure_watch()
    except Exception:  # noqa: BLE001
        log.exception("watch renewal failed")


async def requeue_stuck(settings: Settings) -> None:
    async with sessionmaker_for()() as session:
        n = await queue.requeue_stuck(session, settings.job_stuck_timeout_s)
        await session.commit()
    if n:
        log.warning("requeued %d job(s) stuck in running", n)


async def retry_dead() -> None:
    async with sessionmaker_for()() as session:
        n = await queue.retry_dead(session)
        await session.commit()
    if n:
        log.info("returned %d dead-lettered job(s) to the queue", n)


async def prune_fingerprints() -> None:
    async with sessionmaker_for()() as session:
        n = await repo.prune_fingerprints(session)
        await session.commit()
    log.info("cleared %d stale fingerprint(s)", n)


async def weekly_eval(settings: Settings) -> None:
    """Score the current (model, prompt) against the golden set.

    Imported lazily: eval pulls in scoring code the daemon otherwise never
    needs, and a missing golden.jsonl must not take the scheduler down.
    """
    try:
        from eval.run_eval import run_and_report
    except ImportError:
        log.debug("eval package not importable from this process; skipping")
        return
    try:
        report = await run_and_report(settings=settings)
    except FileNotFoundError:
        log.info("no golden.jsonl yet; skipping weekly eval")
        return
    log.info(
        "weekly eval %s/%s: important-recall=%.3f accuracy=%.3f",
        settings.vllm_model_id,
        settings.prompt_version,
        report["important_recall"],
        report["category_accuracy"],
    )


def build_scheduler(daemon, settings: Settings | None = None) -> AsyncIOScheduler:
    settings = settings or get_settings()
    scheduler = AsyncIOScheduler(timezone="UTC")

    scheduler.add_job(
        renew_watch,
        "interval",
        days=1,
        args=[daemon],
        id="renew_watch",
        # Register once at startup too: a daemon that was down past the 7-day
        # expiry has no watch at all until this runs.
        next_run_time=datetime.now(UTC),
        max_instances=1,
    )
    scheduler.add_job(
        reconcile, "interval", hours=1, args=[daemon], id="reconcile", max_instances=1
    )
    scheduler.add_job(
        requeue_stuck, "interval", minutes=5, args=[settings], id="requeue_stuck", max_instances=1
    )
    scheduler.add_job(retry_dead, "interval", hours=1, id="retry_dead", max_instances=1)
    scheduler.add_job(prune_fingerprints, "cron", day_of_week="sun", hour=4, id="prune")
    scheduler.add_job(
        weekly_eval, "cron", day_of_week="sun", hour=5, args=[settings], id="weekly_eval"
    )
    return scheduler


def main() -> None:  # pragma: no cover - normally hosted inside triage-sync
    """Standalone entry point, for running maintenance without the sync daemon."""
    import asyncio

    from triage.ingest.daemon import SyncDaemon

    logging.basicConfig(level=get_settings().log_level)

    async def _run() -> None:
        scheduler = build_scheduler(SyncDaemon())
        scheduler.start()
        await asyncio.Event().wait()

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
