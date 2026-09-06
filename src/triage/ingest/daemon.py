"""The sync service: Pub/Sub pull -> message rows -> enqueued jobs.

Pull, not push. A pull subscription holds an outbound long-poll, so there is no
public HTTPS endpoint to expose and no inbound firewall rule to open. That is
what makes this viable on-prem.

Pub/Sub is at-least-once but not guaranteed-delivery, so it is never the only
path: ``sync_once`` is idempotent and the scheduler runs it hourly against the
stored watermark regardless of what Pub/Sub did. Without that independent
sweep you silently lose emails and find out weeks later.

APScheduler runs inside this process (see scheduler.py) -- one stateful daemon,
not two.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import sys
from datetime import UTC, datetime, timedelta

from triage.config import Settings, get_settings
from triage.db import repo
from triage.db.engine import configure_event_loop_policy, sessionmaker_for
from triage.ingest.auth import ReconnectRequired
from triage.ingest.base import Checkpoint, HistoryTooOldError, to_message_in
from triage.ingest.gmail_api import GmailSource

log = logging.getLogger("triage.sync")

#: Google keeps roughly a week of history. When the watermark falls outside it,
#: resync this far back and re-checkpoint. Overlap is free -- ON CONFLICT DO
#: NOTHING on gmail_id absorbs it.
RESYNC_WINDOW = timedelta(days=8)


class SyncDaemon:
    def __init__(self, settings: Settings | None = None, source: GmailSource | None = None) -> None:
        self.settings = settings or get_settings()
        self.source = source or GmailSource(self.settings)
        self.sessions = sessionmaker_for()
        # Pub/Sub deliveries and the hourly reconciliation both call
        # sync_once; serialising them keeps two passes from claiming the same
        # historyId window.
        self._lock = asyncio.Lock()
        self.needs_reconnect = False

    # -- core sync ---------------------------------------------------------

    async def sync_once(self, *, reconcile: bool = False) -> dict:
        """Pull everything since the stored watermark. Safe to call at any time."""
        async with self._lock:
            async with self.sessions() as session:
                state = await repo.get_sync_state(session)
                await session.commit()

            if not state.history_id:
                # First run: checkpoint at now. Historical mail is the
                # backfill's job, not this one's.
                history_id = await self.source.current_history_id()
                async with self.sessions() as session:
                    await repo.update_sync_state(session, history_id=history_id)
                    await session.commit()
                log.info("initialised watermark at historyId=%s", history_id)
                return {"stored": 0, "duplicates": 0, "history_id": history_id}

            try:
                result = await self._drain(
                    self.source.fetch_since(Checkpoint(history_id=state.history_id))
                )
            except HistoryTooOldError as exc:
                log.warning("%s; falling back to bounded resync", exc)
                since = datetime.now(UTC) - RESYNC_WINDOW
                result = await self._drain(self.source.resync(since))
            except ReconnectRequired as exc:
                # Surfaces in the UI banner. Retrying will not help until a
                # human re-consents.
                self.needs_reconnect = True
                log.error("gmail credentials need reconnecting: %s", exc)
                return {"stored": 0, "duplicates": 0, "error": "reconnect_required"}

            self.needs_reconnect = False
            new_history_id = self.source.last_history_id
            async with self.sessions() as session:
                await repo.update_sync_state(
                    session, history_id=new_history_id, touch_reconcile=reconcile
                )
                await session.commit()

            log.info(
                "sync%s: stored=%d duplicates=%d watermark=%s",
                " (reconcile)" if reconcile else "",
                result["stored"],
                result["duplicates"],
                new_history_id,
            )
            return {**result, "history_id": new_history_id}

    async def _drain(self, stream) -> dict:
        """Persist a stream of parsed messages, one transaction per message.

        One transaction per message rather than per batch: a single unparseable
        message must not roll back the fifty good ones behind it, and the row
        plus its job still commit together.
        """
        stored = duplicates = failed = 0
        async for parsed in stream:
            try:
                async with self.sessions() as session:
                    message, _job_id = await repo.ingest_message(session, to_message_in(parsed))
                    await session.commit()
                if message is None:
                    duplicates += 1
                else:
                    stored += 1
            except Exception:  # noqa: BLE001 - one bad message must not stop the sync
                failed += 1
                log.exception("failed to store gmail_id=%s", parsed.gmail_id)
        return {"stored": stored, "duplicates": duplicates, "failed": failed}

    # -- watch lifecycle ---------------------------------------------------

    async def ensure_watch(self) -> None:
        """Register or renew users.watch(). Expiry is 7 days; renew daily."""
        if not self.settings.pubsub_topic:
            log.info("PUBSUB_TOPIC unset; running in history.list polling mode")
            return
        result = await self.source.watch()
        async with self.sessions() as session:
            await repo.update_sync_state(session, watch_expiry=result["expiry"])
            await session.commit()

    # -- delivery paths ----------------------------------------------------

    async def run_pubsub(self, stop: asyncio.Event) -> None:
        """Long-poll the pull subscription and sync on every delivery.

        The historyId inside the notification is deliberately ignored: the
        stored watermark is the only thing that decides what gets fetched, so a
        duplicate or out-of-order delivery cannot skip a message.
        """
        from google.cloud import pubsub_v1

        subscriber = pubsub_v1.SubscriberClient()
        path = self.settings.pubsub_subscription
        log.info("pulling from %s", path)

        while not stop.is_set():
            try:
                response = await asyncio.to_thread(
                    subscriber.pull,
                    request={"subscription": path, "max_messages": 10},
                    timeout=60.0,
                )
            except Exception as exc:  # noqa: BLE001 - transient pull failures
                log.warning("pubsub pull failed (%s); retrying in 10s", exc)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=10)
                continue

            if not response.received_messages:
                continue

            for received in response.received_messages:
                with contextlib.suppress(Exception):
                    payload = json.loads(received.message.data.decode("utf-8"))
                    log.debug("notification for %s", payload.get("emailAddress"))

            await self.sync_once()
            # Ack only after the sync committed. An unacked notification is
            # redelivered, which is harmless; a lost one is not.
            await asyncio.to_thread(
                subscriber.acknowledge,
                request={
                    "subscription": path,
                    "ack_ids": [m.ack_id for m in response.received_messages],
                },
            )

    async def run_polling(self, stop: asyncio.Event) -> None:
        """Fallback when Pub/Sub is not configured."""
        interval = self.settings.history_poll_interval_s
        log.info("polling history.list every %ds", interval)
        while not stop.is_set():
            try:
                await self.sync_once()
            except Exception:  # noqa: BLE001
                log.exception("poll failed")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval)

    async def close(self) -> None:
        await self.source.close()


async def run(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    daemon = SyncDaemon(settings)
    stop = asyncio.Event()
    _install_signal_handlers(stop)

    # Imported here so that importing the daemon does not pull APScheduler into
    # processes that do not schedule anything.
    from triage.scheduler import build_scheduler

    scheduler = build_scheduler(daemon)
    scheduler.start()

    try:
        await daemon.ensure_watch()
    except Exception:  # noqa: BLE001 - a failed watch must not stop ingestion
        log.exception("watch registration failed; the hourly sweep still runs")

    delivery = (
        daemon.run_pubsub(stop) if settings.pubsub_subscription else daemon.run_polling(stop)
    )
    try:
        await delivery
    finally:
        scheduler.shutdown(wait=False)
        await daemon.close()


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - Windows dev boxes
            signal.signal(sig, lambda *_: stop.set())


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        stream=sys.stdout,
    )
    configure_event_loop_policy()
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
