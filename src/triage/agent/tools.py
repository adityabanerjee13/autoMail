"""The fifteen things the agent can do, and the code that does them.

Every executor wraps something the console already drives, so the agent and the
buttons on /queue are two front ends over one implementation. In particular the
queue tools drive the *same* ``runner`` singleton, which is why "stop" from the
console stops a drain the agent started. One runner, one truth.

Two rules hold across the whole registry:

* **A tool reads only the fields its own arg model declares, and ignores the
  rest.** An 8B model will put ``limit`` on ``read_message``. Rejecting that
  call teaches it nothing and burns one of six iterations; ignoring the stray
  field costs nothing and gets the user their answer.
* **Executors own their transaction.** Nothing in ``repo`` or ``queue``
  commits, exactly as in the ops routes, so every writing tool commits for
  itself.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from triage.agent.errors import ToolErrorCode, ToolFailure
from triage.agent.format import render_message, render_rows, render_unprocessed
from triage.agent.schema import StepArgs, ToolName
from triage.config import get_settings
from triage.db import queue, repo
from triage.ingest import auth

log = logging.getLogger("triage.agent.tools")

DEFAULT_LIMIT = 10


# ---------------------------------------------------------------------------
# per-tool argument models
#
# Every field name here must also exist on StepArgs -- that is what keeps the
# emitted grammar flat, and there is a test that fails when it stops being
# true. These models are the second gate: the grammar guarantees types and
# bounds, these guarantee a tool got the arguments it actually needs.
# ---------------------------------------------------------------------------


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")


class FindMessagesArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str | None = None
    category: str | None = None
    important_only: bool | None = None
    days: int | None = None
    limit: int = DEFAULT_LIMIT


class MessageIdArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message_id: int


class LimitArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    limit: int = DEFAULT_LIMIT


class OptionalLimitArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    limit: int | None = None


class DaysArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    days: int = 7


class AnswerArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolResult:
    #: What the model sees. Compact, capped, and instructive when truncated.
    text: str
    #: One line for the chat transcript's chip, read by a person.
    summary: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: ToolName
    description: str
    args_model: type[BaseModel]
    run: Callable[[AsyncSession, Any], Awaitable[ToolResult]]
    destructive: bool = False
    timeout_s: float = 10.0
    #: Shown in the confirmation banner. Required when destructive, and it must
    #: state the real consequence -- "messages are kept" is the thing a user
    #: needs to know before clicking yes.
    confirm_prompt: str | None = None


# ---------------------------------------------------------------------------
# read executors
# ---------------------------------------------------------------------------


async def _find_messages(session: AsyncSession, a: FindMessagesArgs) -> ToolResult:
    since = datetime.now(UTC) - timedelta(days=a.days) if a.days else None
    rows = await repo.search_messages(
        session,
        query=a.query,
        category=a.category,
        important=True if a.important_only else None,
        since=since,
        limit=a.limit,
    )
    bits = [b for b in (a.query, a.category, "important" if a.important_only else None) if b]
    what = ", ".join(bits) or "recent mail"

    body = render_rows(rows)
    if not rows and (a.important_only or a.category):
        # Importance and category live on classifications, so on an unclassified
        # mailbox these filters match nothing and "no messages matched" is
        # actively misleading -- it reads as "you have no important mail" rather
        # than "nothing has been judged yet". Observed making the model abandon
        # the filter and read messages one at a time until it ran out of steps.
        unclassified = await repo.count_unprocessed(session)
        if unclassified:
            body = (
                f"no messages matched, because {unclassified} email(s) have not been "
                "classified yet - importance and category only exist once an email "
                "has been classified. Tell the user they need to run the queue "
                "first; do not try to work it out by reading emails one by one."
            )

    return ToolResult(
        text=body,
        summary=f"found {len(rows)} for {what}",
        meta={"count": len(rows), "ids": [r.message.id for r in rows]},
    )


async def _read_message(session: AsyncSession, a: MessageIdArgs) -> ToolResult:
    message = await repo.get_message(session, a.message_id)
    if message is None:
        raise ToolFailure(
            ToolErrorCode.NOT_FOUND,
            f"there is no email with id {a.message_id}. "
            "Call find_messages to see which ids exist.",
        )
    classification = await repo.latest_classification(session, a.message_id)
    return ToolResult(
        text=render_message(message, classification),
        summary=f"read #{message.id}: {message.subject or '(no subject)'}"[:120],
        meta={"message_id": message.id},
    )


async def _list_unprocessed(session: AsyncSession, a: LimitArgs) -> ToolResult:
    rows = await repo.unprocessed(session, limit=a.limit)
    total = await repo.count_unprocessed(session)
    body = render_unprocessed(rows)
    if rows and total > len(rows):
        body = f"{len(rows)} shown of {total} unprocessed\n{body}"
    return ToolResult(text=body, summary=f"{total} unprocessed", meta={"total": total})


async def _list_review_queue(session: AsyncSession, a: LimitArgs) -> ToolResult:
    rows = await repo.review_queue(session, limit=a.limit)
    return ToolResult(
        text=render_rows(rows) if rows else "the review queue is empty.",
        summary=f"{len(rows)} awaiting review",
        meta={"count": len(rows)},
    )


async def _mailbox_stats(session: AsyncSession, a: NoArgs) -> ToolResult:
    s = await repo.stats(session)
    cats = ", ".join(f"{k} {v}" for k, v in sorted(s["by_category"].items())) or "none"
    return ToolResult(
        text=(
            f"{s['messages']} emails, {s['classified']} classified, "
            f"{s['important']} important.\nby category: {cats}"
        ),
        summary=f"{s['messages']} emails, {s['classified']} classified",
        meta={"messages": s["messages"], "classified": s["classified"]},
    )


async def _queue_status(session: AsyncSession, a: NoArgs) -> ToolResult:
    from triage.api.tasks import runner

    counts = await queue.counts(session)
    unprocessed = await repo.count_unprocessed(session)
    st = runner.status()
    settings = get_settings()
    imap = auth.imap_credentials(settings)
    connected = auth.status(settings).connected or bool(imap)
    account = auth.account_email(settings) or (imap[0] if imap else None)

    if st["running"]:
        state = "the queue is running now"
        if st["current_message_id"]:
            state += f", on message {st['current_message_id']}"
        if st["stopping"]:
            state += " (stopping after this one)"
        state += f"; {st['processed']} classified so far this run"
    else:
        state = "the queue is not running"
    lines = [
        " / ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no jobs at all",
        state,
        f"{unprocessed} email(s) have no classification yet",
        f"gmail {'connected' if connected else 'NOT connected'}"
        + (f" as {account}" if account else ""),
    ]
    if st["last_error"]:
        lines.append(f"last error: {st['last_error'][:200]}")
    return ToolResult(
        text="\n".join(lines),
        summary=f"pending {counts.get('pending', 0)}, {unprocessed} unclassified",
        meta={"counts": counts, "unprocessed": unprocessed, "running": st["running"]},
    )


# ---------------------------------------------------------------------------
# write executors
# ---------------------------------------------------------------------------


async def _sync_mail(session: AsyncSession, a: DaysArgs) -> ToolResult:
    """Fetch new mail. Dispatches the way the console's two buttons do.

    An App Password means IMAP, and that is this box's live path; otherwise the
    OAuth history watermark. Not a preference -- with only IMAP credentials
    stored the API path has nothing to authenticate with, and vice versa.
    """
    settings = get_settings()
    if auth.imap_credentials(settings):
        from triage.ingest.imap import sync_recent

        r = await sync_recent(days=a.days, settings=settings)
        body = (
            f"synced over IMAP: {r['stored']} new, {r['duplicates']} already stored, "
            f"{r['failed']} failed, {r['scanned']} scanned from the last {a.days} days."
        )
        stored = r["stored"]
    else:
        from triage.api.tasks import sync_daemon

        r = await sync_daemon.sync_once()
        # The first-run branch returns no 'failed'/'scanned' keys at all, so
        # every read here is a .get().
        stored = r.get("stored", 0)
        body = (
            f"synced over the Gmail API: {stored} new, "
            f"{r.get('duplicates', 0)} already stored."
        )
        if r.get("error") == "reconnect_required":
            body = "the stored Gmail credentials need reconnecting; nothing was synced."
    body += " The new mail is queued but not classified yet."
    return ToolResult(text=body, summary=f"synced: {stored} new", meta=dict(r))


async def _enqueue_unprocessed(session: AsyncSession, a: OptionalLimitArgs) -> ToolResult:
    n = await repo.enqueue_unprocessed(session, limit=a.limit or 10_000)
    await session.commit()
    return ToolResult(
        text=f"queued {n} email(s). Nothing is classifying yet.",
        summary=f"queued {n}",
        meta={"queued": n},
    )


async def _enqueue_message(session: AsyncSession, a: MessageIdArgs) -> ToolResult:
    message = await repo.get_message(session, a.message_id)
    if message is None:
        raise ToolFailure(
            ToolErrorCode.NOT_FOUND,
            f"there is no email with id {a.message_id}. "
            "Call find_messages to see which ids exist.",
        )
    # Reuse an existing pending job rather than stacking a second one, exactly
    # as the console's Process button does.
    job_id = await queue.pending_job_for(session, a.message_id)
    reused = job_id is not None
    if job_id is None:
        job_id = await queue.enqueue(session, a.message_id)
    await session.commit()
    return ToolResult(
        text=(
            f"email {a.message_id} is "
            + ("already queued" if reused else "now queued")
            + " and will be classified next time the queue runs."
        ),
        summary=f"queued #{a.message_id}",
        meta={"job_id": job_id, "reused": reused},
    )


async def _run_queue(session: AsyncSession, a: OptionalLimitArgs) -> ToolResult:
    from triage.api.tasks import runner

    # The limit goes to BOTH calls. Passing it only to the runner would leave
    # every remaining message sitting queued after "classify the newest 20" --
    # the queue depth would say thousands and the user asked for twenty.
    n = await repo.enqueue_unprocessed(session, limit=a.limit or 10_000)
    await session.commit()
    if not runner.start(limit=a.limit):
        st = runner.status()
        return ToolResult(
            text=(
                f"queued {n} more, but the queue was already running "
                f"({st['processed']} classified so far), so nothing new was started. "
                "The messages just queued will be picked up by the run in progress."
            ),
            summary=f"already running, +{n} queued",
            meta={"queued": n, "started": False},
        )
    return ToolResult(
        text=(
            f"queued {n} email(s) and started classifying in the background. "
            "This takes a while; call queue_status to see progress."
        ),
        summary=f"running, {n} queued",
        meta={"queued": n, "started": True},
    )


async def _stop_queue(session: AsyncSession, a: NoArgs) -> ToolResult:
    from triage.api.tasks import runner

    if not runner.running:
        return ToolResult(
            text="the queue was not running, so there was nothing to stop.",
            summary="not running",
            meta={"stopped": False},
        )
    runner.stop()
    return ToolResult(
        text=(
            "stopping. The email being classified right now will finish first - "
            "there is no way to interrupt one mid-classification. Everything else "
            "stays queued."
        ),
        summary="stopping",
        meta={"stopped": True},
    )


async def _retry_dead_jobs(session: AsyncSession, a: LimitArgs) -> ToolResult:
    n = await queue.retry_dead(session, limit=a.limit)
    await session.commit()
    return ToolResult(
        text=f"put {n} failed job(s) back in the queue.",
        summary=f"retried {n}",
        meta={"retried": n},
    )


async def _clear_queue(session: AsyncSession, a: NoArgs) -> ToolResult:
    n = await repo.clear_pending_jobs(session)
    await session.commit()
    log.warning("agent cleared %d pending job(s)", n)
    return ToolResult(
        text=(
            f"deleted {n} queued job(s). The emails themselves are untouched and "
            "can be queued again."
        ),
        summary=f"cleared {n}",
        meta={"cleared": n},
    )


async def _dequeue_message(session: AsyncSession, a: MessageIdArgs) -> ToolResult:
    n = await queue.remove_pending_for(session, a.message_id)
    await session.commit()
    return ToolResult(
        text=(
            f"removed {n} queued job(s) for email {a.message_id}. The email is kept."
            if n
            else f"email {a.message_id} was not queued, so nothing changed."
        ),
        summary=f"dequeued #{a.message_id}",
        meta={"removed": n},
    )


async def _answer(session: AsyncSession, a: AnswerArgs) -> ToolResult:  # pragma: no cover
    """Never executed. The loop intercepts ``answer`` and ends the turn."""
    return ToolResult(text=a.text, summary="answered")


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

REGISTRY: dict[ToolName, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            ToolName.FIND_MESSAGES,
            "List stored emails, newest first. Optional `query` matches the subject, "
            "the sender address or the body. Optional `category` and `important_only` "
            "narrow it, and `days` limits it to the last N days.",
            FindMessagesArgs,
            _find_messages,
        ),
        ToolSpec(
            ToolName.READ_MESSAGE,
            "Read one email in full: sender, date, body, and its current category and "
            "importance. Needs the id from find_messages.",
            MessageIdArgs,
            _read_message,
        ),
        ToolSpec(
            ToolName.LIST_UNPROCESSED,
            "List emails that have not been classified yet, with their queue state.",
            LimitArgs,
            _list_unprocessed,
        ),
        ToolSpec(
            ToolName.LIST_REVIEW_QUEUE,
            "List emails that were classified with low confidence and need a human to "
            "check them.",
            LimitArgs,
            _list_review_queue,
        ),
        ToolSpec(
            ToolName.MAILBOX_STATS,
            "Counts for the whole mailbox: how many emails, how many classified, how "
            "many important, and the breakdown by category.",
            NoArgs,
            _mailbox_stats,
        ),
        ToolSpec(
            ToolName.QUEUE_STATUS,
            "The state of the classification queue: how many jobs are waiting, running, "
            "done and failed, whether it is classifying right now, and whether Gmail is "
            "connected.",
            NoArgs,
            _queue_status,
        ),
        ToolSpec(
            ToolName.SYNC_MAIL,
            "Fetch new email from Gmail into the database. `days` is how far back to "
            "look, default 7. This does not classify anything.",
            DaysArgs,
            _sync_mail,
            timeout_s=180.0,
        ),
        ToolSpec(
            ToolName.ENQUEUE_UNPROCESSED,
            "Put unclassified emails into the queue without starting it. Use run_queue "
            "instead if the user wants them classified now.",
            OptionalLimitArgs,
            _enqueue_unprocessed,
            timeout_s=20.0,
        ),
        ToolSpec(
            ToolName.ENQUEUE_MESSAGE,
            "ADD one specific email TO the queue so it gets classified. Use this "
            "whenever the user wants an email queued, re-queued, or put back in.",
            MessageIdArgs,
            _enqueue_message,
            timeout_s=20.0,
        ),
        ToolSpec(
            ToolName.RUN_QUEUE,
            "Queue every unclassified email and start classifying now, in the "
            "background. Returns immediately - use queue_status to check progress. "
            "`limit` classifies only that many.",
            OptionalLimitArgs,
            _run_queue,
        ),
        ToolSpec(
            ToolName.STOP_QUEUE,
            "Stop classifying after the email currently in flight finishes. Queued "
            "emails stay queued.",
            NoArgs,
            _stop_queue,
        ),
        ToolSpec(
            ToolName.RETRY_DEAD_JOBS,
            "Put failed classification jobs back into the queue to try again.",
            LimitArgs,
            _retry_dead_jobs,
            timeout_s=20.0,
        ),
        ToolSpec(
            ToolName.CLEAR_QUEUE,
            "Delete every queued classification job. The emails are kept and can be "
            "queued again.",
            NoArgs,
            _clear_queue,
            destructive=True,
            timeout_s=20.0,
            confirm_prompt=(
                "Delete every queued classification job? The emails themselves are "
                "kept and can be queued again, and anything already classifying is "
                "left alone."
            ),
        ),
        ToolSpec(
            ToolName.DEQUEUE_MESSAGE,
            "REMOVE one email FROM the queue so it will NOT be classified. This "
            "only cancels queued work - it never adds any. The email is kept.",
            MessageIdArgs,
            _dequeue_message,
            destructive=True,
            timeout_s=20.0,
            confirm_prompt=(
                "Remove this email from the queue so it is not classified? The "
                "email itself is kept."
            ),
        ),
        ToolSpec(
            ToolName.ANSWER,
            "Reply to the user and end your turn. Use this as soon as you can answer "
            "the question, and always as your last step.",
            AnswerArgs,
            _answer,
        ),
    )
}


def coerce_args(spec: ToolSpec, args: StepArgs) -> BaseModel:
    """Narrow the flat StepArgs to the tool's own model.

    ``exclude_none`` is what implements the ignore-strays rule: a null field the
    model did not set never reaches the tool, so a required argument that was
    genuinely omitted still fails, while a stray ``limit`` on ``read_message``
    is simply dropped by ``extra="ignore"``.
    """
    return spec.args_model.model_validate(args.model_dump(exclude_none=True))


async def sweep_interrupted(session: AsyncSession) -> int:
    """Fail any chat rows a restart left mid-flight.

    The confirmation futures and the turn tasks live in this process, so a
    restart strands every ``running`` and ``awaiting_confirm`` row: nothing will
    ever resolve them and the page would poll forever. Same shape and same
    reason as ``queue.requeue_stuck``.
    """
    result = await session.execute(
        sql_text(
            """
            UPDATE chat_messages
               SET status = 'error', error_code = 'failed',
                   content = 'interrupted by a restart'
             WHERE status IN ('running', 'awaiting_confirm')
            """
        )
    )
    return result.rowcount or 0
