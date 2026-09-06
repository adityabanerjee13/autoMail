"""Per-process tracking for in-flight chat turns.

The same discipline as ``QueueRunner``: a detached asyncio task, a dict of what
is running, and a status the page can poll. Instantiated once in
``api/tasks.py`` so every module singleton lives in one file.

``start`` is exclusive **globally**, not per thread. Two simultaneous turns
would be two ~6,300-token prompts against a vLLM with a 0.55 memory fraction on
an iGPU, which is the one contention case that reliably causes preemption and
recompute -- and slows down the classification the user is watching in the
console at the same time. On a single-user box there is no legitimate reason
for two turns at once.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("triage.agent.runner")


class AgentRunner:
    def __init__(self) -> None:
        self._tasks: dict[int, asyncio.Task] = {}
        #: chat_message_id -> the future a confirm/cancel POST resolves.
        self._pending: dict[int, asyncio.Future[bool]] = {}
        self.last_error: dict[int, str] = {}

    # -- state -------------------------------------------------------------

    def is_running(self, thread_id: int) -> bool:
        task = self._tasks.get(thread_id)
        return task is not None and not task.done()

    def is_busy(self) -> bool:
        return any(not t.done() for t in self._tasks.values())

    def busy_thread(self) -> int | None:
        for thread_id, task in self._tasks.items():
            if not task.done():
                return thread_id
        return None

    def status(self, thread_id: int) -> dict[str, Any]:
        return {
            "running": self.is_running(thread_id),
            "awaiting_confirm": self.awaiting(thread_id),
            "last_error": self.last_error.get(thread_id),
        }

    def awaiting(self, thread_id: int) -> int | None:
        """The chat_message id this thread is blocked on, if any.

        Only meaningful for the running thread; the durable answer is the row's
        own ``awaiting_confirm`` status, which is what survives a restart.
        """
        if not self.is_running(thread_id):
            return None
        for message_id, fut in self._pending.items():
            if not fut.done():
                return message_id
        return None

    # -- lifecycle ---------------------------------------------------------

    def start(self, thread_id: int, coro) -> bool:
        """Run one turn. False if any turn is already in flight.

        The caller builds the coroutine, so this module needs to know nothing
        about the loop or the database.
        """
        if self.is_busy():
            coro.close()  # never awaited; closing avoids "never retrieved"
            return False
        self.last_error.pop(thread_id, None)
        task = asyncio.create_task(coro, name=f"agent-turn-{thread_id}")
        self._tasks[thread_id] = task
        task.add_done_callback(lambda t: self._finish(thread_id, t))
        return True

    def _finish(self, thread_id: int, task: asyncio.Task) -> None:
        self._tasks.pop(thread_id, None)
        # Any confirmation still outstanding belongs to a turn that is over.
        for message_id, fut in list(self._pending.items()):
            if not fut.done():
                fut.cancel()
            self._pending.pop(message_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.exception("agent turn for thread %s failed", thread_id, exc_info=exc)
            self.last_error[thread_id] = repr(exc)

    def cancel(self, thread_id: int) -> bool:
        task = self._tasks.get(thread_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    # -- confirmation ------------------------------------------------------

    async def wait_for_confirmation(self, message_id: int) -> bool:
        """Block the turn until the user answers. Awaited from inside the loop."""
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[message_id] = fut
        try:
            return await fut
        finally:
            self._pending.pop(message_id, None)

    def resolve_confirmation(self, message_id: int, approved: bool) -> bool:
        """Answer a pending confirmation. False if there is nothing to answer.

        A double-click, a stale page, or a confirmation left over from before a
        restart all land here as ``False``, which the route turns into a muted
        "that has already been answered" rather than a 500.
        """
        fut = self._pending.get(message_id)
        if fut is None or fut.done():
            return False
        fut.set_result(approved)
        return True
