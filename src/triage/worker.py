"""Triage worker: LISTEN for work, drain the queue, run the pipeline.

Run 2-4 of these. Each holds one dedicated LISTEN connection and caps its own
in-flight vLLM requests with a semaphore, so total concurrency against the GPU
is WORKER_CONCURRENCY x process count. Size that against the KV cache budget,
not against CPU count.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys

from triage.config import Settings, get_settings
from triage.db import queue
from triage.db.engine import configure_event_loop_policy, sessionmaker_for
from triage.llm.client import LLMClient
from triage.pipeline.runner import run_job

log = logging.getLogger("triage.worker")

#: Idle wake interval. This is not polling for new work -- NOTIFY covers that.
#: It is how a job whose retry backoff has expired becomes runnable again,
#: since nothing issues a NOTIFY at run_after.
IDLE_WAKE_S = 30


async def _listener(wake: asyncio.Event, stop: asyncio.Event) -> None:
    """Set the wake event on every NOTIFY, reconnecting if the socket drops."""
    while not stop.is_set():
        try:
            async for _payload in queue.listen(stop):
                wake.set()
                if stop.is_set():
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dropped LISTEN must not kill the worker
            log.warning("LISTEN connection lost (%s); reconnecting in 5s", exc)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=5)


async def _heartbeat(wake: asyncio.Event, stop: asyncio.Event) -> None:
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=IDLE_WAKE_S)
        wake.set()


async def _run_one(sessions, job, client, sem: asyncio.Semaphore) -> None:
    try:
        outcome = await run_job(sessions, job, client=client)
        log.info(
            "job=%s message=%s status=%s%s",
            job.id,
            job.message_id,
            outcome.status,
            " needs_review" if outcome.needs_review else "",
        )
    finally:
        sem.release()


async def _drain(sessions, client, sem: asyncio.Semaphore, tasks: set) -> int:
    """Claim and dispatch until the queue is empty or concurrency is saturated."""
    dispatched = 0
    while True:
        await sem.acquire()
        async with sessions() as session:
            job = await queue.claim_one(session)
            await session.commit()
        if job is None:
            sem.release()
            return dispatched
        task = asyncio.create_task(_run_one(sessions, job, client, sem))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        dispatched += 1


async def run_worker(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    sessions = sessionmaker_for()
    client = LLMClient(settings)
    sem = asyncio.Semaphore(settings.worker_concurrency)
    tasks: set[asyncio.Task] = set()

    stop = asyncio.Event()
    wake = asyncio.Event()
    wake.set()  # drain anything left over from a previous run before blocking

    _install_signal_handlers(stop)

    listener = asyncio.create_task(_listener(wake, stop))
    heartbeat = asyncio.create_task(_heartbeat(wake, stop))
    log.info(
        "worker %s up: concurrency=%d model=%s prompt=%s",
        os.getpid(),
        settings.worker_concurrency,
        settings.vllm_model_id,
        settings.prompt_version,
    )

    try:
        while not stop.is_set():
            await wake.wait()
            wake.clear()
            n = await _drain(sessions, client, sem, tasks)
            if n:
                log.debug("dispatched %d job(s)", n)
    finally:
        log.info("worker draining %d in-flight job(s)", len(tasks))
        for t in (listener, heartbeat):
            t.cancel()
        if tasks:
            # Let claimed jobs finish rather than leaving rows in 'running' for
            # the stuck sweep to clean up.
            await asyncio.wait(tasks, timeout=settings.job_stuck_timeout_s)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(listener, heartbeat, return_exceptions=True)


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows dev boxes: no add_signal_handler on the proactor loop.
            signal.signal(sig, lambda *_: stop.set())


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        stream=sys.stdout,
    )
    configure_event_loop_policy()
    asyncio.run(run_worker(settings))


if __name__ == "__main__":
    main()
