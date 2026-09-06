"""Background work the console can start: draining the queue, and OAuth consent.

The queue runner here is not a second implementation of the worker. It claims
from the same ``jobs`` table with the same ``claim_one`` and runs the same
``run_job``, so a standalone ``triage-worker`` process and this in-API runner
can be up at once and ``FOR UPDATE SKIP LOCKED`` keeps them from colliding.
What it adds is a button: on a single-user box you should not have to open a
terminal to get one message classified.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from typing import Any

from triage.agent.runner import AgentRunner
from triage.config import get_settings
from triage.db import queue
from triage.db.engine import sessionmaker_for
from triage.ingest.daemon import SyncDaemon
from triage.llm.client import LLMClient
from triage.pipeline.runner import run_job

log = logging.getLogger("triage.api.tasks")


class QueueRunner:
    """Drains the queue until it is empty, then stops.

    Deliberately not a daemon: it runs when asked and finishes when the work
    is gone, so the console can show "idle" and mean it. For continuous
    operation, run `triage-worker` -- that is what LISTEN/NOTIFY is for.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.processed = 0
        self.failed = 0
        self.skipped = 0
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.last_error: str | None = None
        self.current: int | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "stopping": self.stopping,
            "processed": self.processed,
            "failed": self.failed,
            "skipped": self.skipped,
            "current_message_id": self.current,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "last_error": self.last_error,
        }

    def _begin(self) -> None:
        self._stop.clear()
        self.processed = self.failed = self.skipped = 0
        self.last_error = None
        self.finished_at = None
        self.started_at = datetime.now(UTC)

    def start(self, *, limit: int | None = None) -> bool:
        """Drain the whole queue. False if something is already in flight."""
        if self.running:
            return False
        self._begin()
        self._task = asyncio.create_task(self._drain(limit))
        return True

    def start_one(self, job_id: int) -> bool:
        """Run exactly one named job.

        Separate from ``start`` on purpose. Reusing the drain here meant that
        "Process" on a single message quietly classified the entire mailbox --
        and, because a drain that had just been stopped got restarted by it, it
        also made the Stop button look broken.
        """
        if self.running:
            return False
        self._begin()
        self._task = asyncio.create_task(self._run_one(job_id))
        return True

    def stop(self) -> None:
        """Ask the run to stop once the message in flight finishes.

        There is no safe way to interrupt a message mid-pipeline -- the LLM call
        is in progress and the classification is not yet written -- so this is
        never instant. `stopping` is what the console shows in the meantime, so
        the wait looks deliberate rather than ignored.
        """
        if self.running:
            self._stop.set()

    @property
    def stopping(self) -> bool:
        return self.running and self._stop.is_set()


    def _record(self, outcome: Any) -> None:
        if outcome.status in ("classified", "deduped"):
            self.processed += 1
        elif outcome.status in ("skipped", "missing"):
            self.skipped += 1
        else:
            self.failed += 1
            self.last_error = outcome.error

    async def _run_one(self, job_id: int) -> None:
        settings = get_settings()
        sessions = sessionmaker_for()
        client = LLMClient(settings)
        try:
            async with sessions() as session:
                job = await queue.claim_job(session, job_id)
                await session.commit()
            if job is None:
                # A worker or an in-flight drain got there first. Not an error:
                # the message still gets classified, just not by this click.
                self.last_error = "job was already claimed by another runner"
                return
            self.current = job.message_id
            self._record(await run_job(sessions, job, client=client, settings=settings))
        except Exception as exc:  # noqa: BLE001 - surfaced in the console
            log.exception("single-job run failed")
            self.last_error = repr(exc)
        finally:
            self.current = None
            self.finished_at = datetime.now(UTC)

    async def _drain(self, limit: int | None) -> None:
        settings = get_settings()
        sessions = sessionmaker_for()
        client = LLMClient(settings)
        done = 0
        try:
            while not self._stop.is_set():
                async with sessions() as session:
                    job = await queue.claim_one(session)
                    await session.commit()
                if job is None:
                    break

                self.current = job.message_id
                outcome = await run_job(sessions, job, client=client, settings=settings)
                self.current = None

                self._record(outcome)

                done += 1
                if limit is not None and done >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 - surfaced in the console, not raised
            log.exception("queue runner failed")
            self.last_error = repr(exc)
        finally:
            self.current = None
            self.finished_at = datetime.now(UTC)


class GmailConnect:
    """Runs the desktop OAuth consent flow off the request thread.

    ``run_local_server`` opens a browser and blocks until the user consents, so
    it cannot happen inside a request. It also opens that browser *on the
    machine running the API* -- fine here, where the UI is bound to loopback
    and the operator is sitting at the same box.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.started_at: datetime | None = None
        #: The Google consent URL, captured so the page can offer it as a link.
        self.auth_url: str | None = None
        #: Result line from the most recent manual sync, shown in the console.
        self.last_sync: str | None = None

    @property
    def in_progress(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.in_progress:
            return False
        self.error = None
        self.auth_url = None
        self.started_at = datetime.now(UTC)
        self._thread = threading.Thread(target=self._run, daemon=True, name="gmail-consent")
        self._thread.start()
        return True

    def _run(self) -> None:
        import webbrowser

        from triage.ingest.auth import run_consent_flow

        # run_local_server opens the browser itself and does not hand back the
        # URL, which is a problem when it cannot open one -- a headless box, a
        # service account, an RDP session. Borrowing webbrowser.open for the
        # duration captures the URL so the page can show it as a link, and
        # still opens the browser when that works. Narrow and short-lived: the
        # in_progress guard means only one of these runs at a time.
        real_open = webbrowser.open

        def capture(url: str, *args: Any, **kwargs: Any) -> bool:
            self.auth_url = url
            try:
                return real_open(url, *args, **kwargs)
            except Exception:  # noqa: BLE001 - no browser is not a failure here
                return False

        webbrowser.open = capture  # type: ignore[assignment]
        try:
            run_consent_flow()
            self.auth_url = None
            log.info("gmail consent completed")
        except Exception as exc:  # noqa: BLE001 - reported in the UI
            log.warning("gmail consent failed: %s", exc)
            self.error = str(exc)
        finally:
            webbrowser.open = real_open  # type: ignore[assignment]


#: One of each per API process.
runner = QueueRunner()
gmail_connect = GmailConnect()
agent_runner = AgentRunner()

#: One SyncDaemon, not one per request.
#
# Its __init__ builds an asyncio.Lock whose docstring promises to serialise
# Pub/Sub delivery against the reconciliation sweep -- but /gmail/sync
# constructed a fresh daemon on every call, so each got its own lock and the
# promise held only against itself. Two overlapping sync_once() calls read the
# same historyId watermark and drain the same window; ON CONFLICT DO NOTHING
# makes that harmless to the data and wasteful of a Gmail round trip. Sharing
# one instance makes the lock mean what it says. The agent's sync_mail tool
# uses this too, which is what makes it worth fixing now: a chat request and a
# console click are much likelier to overlap than two console clicks.
sync_daemon = SyncDaemon()
