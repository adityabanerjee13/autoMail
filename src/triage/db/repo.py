"""Every query in the system, returning Pydantic models rather than ORM rows.

If a SQLAlchemy object escapes this module, worker code eventually touches a
detached instance and you get a lazy-load error at 3am inside a retry loop.
Convert at the boundary, always.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Select, and_, desc, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from triage.config import get_settings
from triage.db import queue
from triage.db.models import ClassificationRow, JobRow, MessageRow, SyncStateRow
from triage.schemas import (
    Classification,
    ClassificationIn,
    Message,
    MessageIn,
    MessageWithClassification,
    SyncState,
)

# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


def _to_message(row: MessageRow) -> Message:
    return Message(
        id=row.id,
        gmail_id=row.gmail_id,
        thread_id=row.thread_id,
        message_id_hdr=row.message_id_hdr,
        from_addr=row.from_addr,
        to_addrs=list(row.to_addrs or []),
        cc_addrs=list(row.cc_addrs or []),
        subject=row.subject,
        body_clean=row.body_clean or "",
        snippet=row.snippet,
        labels=list(row.labels or []),
        headers=dict(row.headers or {}),
        fingerprint=row.fingerprint,
        internal_date=row.internal_date,
        created_at=row.created_at,
    )


async def insert_message(session: AsyncSession, msg: MessageIn) -> Message | None:
    """Insert, or return None if this gmail_id is already stored.

    ON CONFLICT DO NOTHING is what makes re-ingestion after a stale watermark
    safe: replaying a week of history produces no duplicates and no errors.
    """
    stmt = (
        pg_insert(MessageRow)
        .values(
            gmail_id=msg.gmail_id,
            thread_id=msg.thread_id,
            message_id_hdr=msg.message_id_hdr,
            from_addr=msg.from_addr,
            to_addrs=msg.to_addrs,
            cc_addrs=msg.cc_addrs,
            subject=msg.subject,
            body_clean=msg.body_clean,
            snippet=msg.snippet,
            labels=msg.labels,
            headers=msg.headers,
            internal_date=msg.internal_date,
        )
        .on_conflict_do_nothing(index_elements=[MessageRow.gmail_id])
        .returning(MessageRow)
    )
    row = (await session.execute(stmt)).scalars().first()
    return _to_message(row) if row is not None else None


async def ingest_message(
    session: AsyncSession, msg: MessageIn
) -> tuple[Message | None, int | None]:
    """Store a message and enqueue its triage job in a single transaction.

    Returns (None, None) when the message was already present, so callers can
    count duplicates without a second round trip.
    """
    stored = await insert_message(session, msg)
    if stored is None:
        return None, None
    job_id = await queue.enqueue(session, stored.id)
    return stored, job_id


async def get_message(session: AsyncSession, message_id: int) -> Message | None:
    row = await session.get(MessageRow, message_id)
    return _to_message(row) if row else None


async def get_message_by_gmail_id(session: AsyncSession, gmail_id: str) -> Message | None:
    row = (
        await session.execute(select(MessageRow).where(MessageRow.gmail_id == gmail_id))
    ).scalar_one_or_none()
    return _to_message(row) if row else None


async def existing_gmail_ids(session: AsyncSession, gmail_ids: list[str]) -> set[str]:
    """Bulk existence check, so the backfill can skip fetching known bodies."""
    if not gmail_ids:
        return set()
    result = await session.execute(
        select(MessageRow.gmail_id).where(MessageRow.gmail_id.in_(gmail_ids))
    )
    return set(result.scalars().all())


async def thread_context(
    session: AsyncSession, thread_id: str, *, exclude_id: int, limit: int = 5
) -> list[dict[str, Any]]:
    """Prior messages in the thread, newest first, as prompt context.

    Snippets only. Full quoted history inflates latency without improving
    accuracy, which is why the prompt never sees bodies from this call.
    """
    result = await session.execute(
        select(
            MessageRow.from_addr,
            MessageRow.subject,
            MessageRow.snippet,
            MessageRow.internal_date,
            MessageRow.labels,
        )
        .where(and_(MessageRow.thread_id == thread_id, MessageRow.id != exclude_id))
        .order_by(desc(MessageRow.internal_date))
        .limit(limit)
    )
    return [
        {
            "from": r.from_addr,
            "subject": r.subject,
            "snippet": (r.snippet or "")[:200],
            "date": r.internal_date.isoformat(),
            "sent_by_user": "SENT" in (r.labels or []),
        }
        for r in result
    ]


async def set_message_derived(
    session: AsyncSession,
    message_id: int,
    *,
    fingerprint: str | None = None,
    embedding: list[float] | None = None,
) -> None:
    """The one sanctioned mutation of a messages row: derived columns only."""
    values: dict[str, Any] = {}
    if fingerprint is not None:
        values["fingerprint"] = fingerprint
    if embedding is not None:
        values["embedding"] = embedding
    if not values:
        return
    await session.execute(update(MessageRow).where(MessageRow.id == message_id).values(**values))


async def prune_fingerprints(session: AsyncSession, older_than_days: int = 180) -> int:
    """Clear fingerprints on old messages to bound the dedup lookup.

    The classification rows stay -- only the dedup key is dropped.
    """
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
    result = await session.execute(
        update(MessageRow)
        .where(and_(MessageRow.internal_date < cutoff, MessageRow.fingerprint.isnot(None)))
        .values(fingerprint=None)
        .returning(MessageRow.id)
    )
    return len(result.all())


# ---------------------------------------------------------------------------
# sender features
# ---------------------------------------------------------------------------


async def sender_features(session: AsyncSession, from_addr: str, *, before: datetime) -> dict:
    """Cheap history signals for the prompt.

    'Have we ever replied to this sender' is the single most useful feature
    here, and it is only available because SENT is synced alongside INBOX.
    """
    total = await session.scalar(
        select(func.count())
        .select_from(MessageRow)
        .where(and_(MessageRow.from_addr == from_addr, MessageRow.internal_date < before))
    )
    replied = await session.scalar(
        select(func.count())
        .select_from(MessageRow)
        .where(
            and_(
                MessageRow.labels.any("SENT"),
                or_(
                    MessageRow.to_addrs.any(from_addr),
                    MessageRow.cc_addrs.any(from_addr),
                ),
            )
        )
    )
    important = await session.scalar(
        select(func.count())
        .select_from(ClassificationRow)
        .join(MessageRow, MessageRow.id == ClassificationRow.message_id)
        .where(
            and_(
                MessageRow.from_addr == from_addr,
                ClassificationRow.is_important.is_(True),
            )
        )
    )
    return {
        "messages_from_sender": int(total or 0),
        "user_has_replied": bool(replied),
        "prior_important_from_sender": int(important or 0),
    }


# ---------------------------------------------------------------------------
# classifications
# ---------------------------------------------------------------------------


def _to_classification(row: ClassificationRow) -> Classification:
    return Classification(
        id=row.id,
        message_id=row.message_id,
        is_important=row.is_important,
        category=row.category,
        payload=dict(row.payload or {}),
        raw_response=row.raw_response,
        model_id=row.model_id,
        prompt_version=row.prompt_version,
        schema_version=row.schema_version,
        source=row.source,  # type: ignore[arg-type]
        input_tokens=row.input_tokens,
        latency_ms=row.latency_ms,
        created_at=row.created_at,
    )


async def insert_classification(session: AsyncSession, c: ClassificationIn) -> Classification:
    row = ClassificationRow(**c.model_dump())
    session.add(row)
    await session.flush()
    return _to_classification(row)


async def latest_classification(
    session: AsyncSession, message_id: int
) -> Classification | None:
    """Latest row wins -- including a human correction appended after the LLM."""
    row = (
        await session.execute(
            select(ClassificationRow)
            .where(ClassificationRow.message_id == message_id)
            .order_by(desc(ClassificationRow.id))
            .limit(1)
        )
    ).scalar_one_or_none()
    return _to_classification(row) if row else None


async def has_classification(
    session: AsyncSession, message_id: int, *, model_id: str, prompt_version: str
) -> bool:
    """Idempotency guard for runner step 2.

    Scoped to (model, prompt) on purpose: a re-run under a new prompt version
    is a new judgment worth spending GPU time on, a re-run under the same one
    is not.
    """
    found = await session.scalar(
        select(ClassificationRow.id)
        .where(
            and_(
                ClassificationRow.message_id == message_id,
                ClassificationRow.model_id == model_id,
                ClassificationRow.prompt_version == prompt_version,
            )
        )
        .limit(1)
    )
    return found is not None


async def dedup_candidate(
    session: AsyncSession,
    fingerprint: str,
    *,
    model_id: str,
    prompt_version: str,
    exclude_message_id: int,
) -> Classification | None:
    """Most recent judgment for a message with the same template fingerprint.

    Restricted to the current (model, prompt) so a copied result is never
    attributed to a labeller that did not produce it. A human correction on a
    template beats an LLM row from the same template.
    """
    row = (
        await session.execute(
            select(ClassificationRow)
            .join(MessageRow, MessageRow.id == ClassificationRow.message_id)
            .where(
                and_(
                    MessageRow.fingerprint == fingerprint,
                    MessageRow.id != exclude_message_id,
                    ClassificationRow.model_id == model_id,
                    ClassificationRow.prompt_version == prompt_version,
                )
            )
            .order_by(
                # human corrections first, then most recent
                (ClassificationRow.source == "human").desc(),
                desc(ClassificationRow.id),
            )
            .limit(1)
        )
    ).scalars().first()
    return _to_classification(row) if row else None


async def nearest_labelled(
    session: AsyncSession,
    embedding: list[float],
    *,
    k: int = 5,
    exclude_message_id: int | None = None,
    prefer_human: bool = True,
) -> list[tuple[Message, Classification]]:
    """pgvector nearest neighbours among already-labelled messages.

    Returns an empty list on day one, which is exactly why ENABLE_FEWSHOT
    defaults to false -- injecting zero examples is a wasted prompt section.
    """
    latest = (
        select(
            ClassificationRow.message_id.label("mid"),
            func.max(ClassificationRow.id).label("cid"),
        )
        .group_by(ClassificationRow.message_id)
        .subquery()
    )
    stmt: Select = (
        select(MessageRow, ClassificationRow)
        .join(latest, latest.c.mid == MessageRow.id)
        .join(ClassificationRow, ClassificationRow.id == latest.c.cid)
        .where(MessageRow.embedding.isnot(None))
        .order_by(MessageRow.embedding.cosine_distance(embedding))
        .limit(k * 3 if prefer_human else k)
    )
    if exclude_message_id is not None:
        stmt = stmt.where(MessageRow.id != exclude_message_id)

    rows = (await session.execute(stmt)).all()
    pairs = [(_to_message(m), _to_classification(c)) for m, c in rows]
    if prefer_human:
        pairs.sort(key=lambda p: 0 if p[1].source == "human" else 1)
    return pairs[:k]


# ---------------------------------------------------------------------------
# review / listing
# ---------------------------------------------------------------------------


def _latest_join() -> Any:
    return (
        select(
            ClassificationRow.message_id.label("mid"),
            func.max(ClassificationRow.id).label("cid"),
        )
        .group_by(ClassificationRow.message_id)
        .subquery()
    )


async def list_messages(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    important: bool | None = None,
    search: str | None = None,
) -> list[MessageWithClassification]:
    latest = _latest_join()
    stmt = (
        select(MessageRow, ClassificationRow)
        .outerjoin(latest, latest.c.mid == MessageRow.id)
        .outerjoin(ClassificationRow, ClassificationRow.id == latest.c.cid)
        .order_by(desc(MessageRow.internal_date), desc(MessageRow.id))
        .limit(limit)
        .offset(offset)
    )
    if category:
        stmt = stmt.where(ClassificationRow.category == category)
    if important is not None:
        stmt = stmt.where(ClassificationRow.is_important.is_(important))
    if search:
        like = f"%{search}%"
        stmt = stmt.where(
            or_(MessageRow.subject.ilike(like), MessageRow.from_addr.ilike(like))
        )

    out: list[MessageWithClassification] = []
    for m, c in (await session.execute(stmt)).all():
        cl = _to_classification(c) if c is not None else None
        out.append(
            MessageWithClassification(
                message=_to_message(m),
                classification=cl,
                needs_review=_needs_review(cl),
            )
        )
    return out


async def search_messages(
    session: AsyncSession,
    *,
    query: str | None = None,
    category: str | None = None,
    important: bool | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[MessageWithClassification]:
    """Search for the agent. Deliberately not a change to ``list_messages``.

    Two differences from the message list, and both exist because a person
    asking a question is not a person scrolling a table:

    * ``query`` also matches ``body_clean``. "the email about the tax return"
      is a body search, and the list view's subject-or-sender ILIKE cannot
      answer it. This is a sequential scan -- there is no trigram index -- which
      costs a couple hundred milliseconds on a personal mailbox and is fine at
      one query per tool call. If it ever stops being fine the fix is
      ``CREATE INDEX ... USING gin (body_clean gin_trgm_ops)``, not a rewrite.
    * date bounds, so "mail from last week" is answerable at all.

    ``list_messages`` keeps its exact behaviour so that ``/`` and its callers
    are untouched by any of this.
    """
    latest = _latest_join()
    stmt = (
        select(MessageRow, ClassificationRow)
        .outerjoin(latest, latest.c.mid == MessageRow.id)
        .outerjoin(ClassificationRow, ClassificationRow.id == latest.c.cid)
        .order_by(desc(MessageRow.internal_date), desc(MessageRow.id))
        .limit(limit)
        .offset(offset)
    )
    if category:
        stmt = stmt.where(ClassificationRow.category == category)
    if important is not None:
        stmt = stmt.where(ClassificationRow.is_important.is_(important))
    if since is not None:
        stmt = stmt.where(MessageRow.internal_date >= since)
    if until is not None:
        stmt = stmt.where(MessageRow.internal_date <= until)
    if query:
        like = f"%{query}%"
        stmt = stmt.where(
            or_(
                MessageRow.subject.ilike(like),
                MessageRow.from_addr.ilike(like),
                MessageRow.body_clean.ilike(like),
            )
        )

    out: list[MessageWithClassification] = []
    for m, c in (await session.execute(stmt)).all():
        cl = _to_classification(c) if c is not None else None
        out.append(
            MessageWithClassification(
                message=_to_message(m),
                classification=cl,
                needs_review=_needs_review(cl),
            )
        )
    return out


def _needs_review(c: Classification | None) -> bool:
    """Low confidence on either axis routes a message to the review queue.

    A human row is never in the queue: the point of the queue is to collect
    human rows.
    """
    if c is None or c.source == "human":
        return False
    p = c.payload or {}
    return "low" in {p.get("importance_confidence"), p.get("category_confidence")}


async def review_queue(
    session: AsyncSession, *, limit: int = 50, offset: int = 0
) -> list[MessageWithClassification]:
    latest = _latest_join()
    stmt = (
        select(MessageRow, ClassificationRow)
        .join(latest, latest.c.mid == MessageRow.id)
        .join(ClassificationRow, ClassificationRow.id == latest.c.cid)
        .where(
            and_(
                ClassificationRow.source != "human",
                or_(
                    ClassificationRow.payload["importance_confidence"].astext == "low",
                    ClassificationRow.payload["category_confidence"].astext == "low",
                ),
            )
        )
        .order_by(desc(MessageRow.internal_date), desc(MessageRow.id))
        .limit(limit)
        .offset(offset)
    )
    return [
        MessageWithClassification(
            message=_to_message(m), classification=_to_classification(c), needs_review=True
        )
        for m, c in (await session.execute(stmt)).all()
    ]


async def stats(session: AsyncSession) -> dict[str, Any]:
    latest = _latest_join()
    by_category = (
        await session.execute(
            select(ClassificationRow.category, func.count())
            .join(latest, latest.c.cid == ClassificationRow.id)
            .group_by(ClassificationRow.category)
            .order_by(desc(func.count()))
        )
    ).all()
    by_source = (
        await session.execute(
            select(ClassificationRow.source, func.count())
            .join(latest, latest.c.cid == ClassificationRow.id)
            .group_by(ClassificationRow.source)
        )
    ).all()
    return {
        "messages": int(await session.scalar(select(func.count()).select_from(MessageRow)) or 0),
        "classified": int(
            await session.scalar(select(func.count()).select_from(latest)) or 0
        ),
        "important": int(
            await session.scalar(
                select(func.count())
                .select_from(ClassificationRow)
                .join(latest, latest.c.cid == ClassificationRow.id)
                .where(ClassificationRow.is_important.is_(True))
            )
            or 0
        ),
        "by_category": {k: v for k, v in by_category},
        "by_source": {k: v for k, v in by_source},
        "jobs": await queue.counts(session),
    }


async def unprocessed(
    session: AsyncSession, *, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    """Messages with no classification at the current model + prompt version.

    "Unprocessed" is deliberately version-scoped rather than "has no
    classification row at all": bumping PROMPT_VERSION is meant to make the
    whole mailbox re-processable, and this listing is where that becomes
    visible. Each row carries its live job status, so the console can tell a
    message nobody has queued from one that is queued, running, or dead.
    """
    settings = get_settings()
    result = await session.execute(
        text(
            """
            SELECT m.id, m.from_addr, m.subject, m.internal_date,
                   j.status AS job_status, j.attempts, j.run_after, j.last_error
              FROM messages m
              LEFT JOIN LATERAL (
                    SELECT status, attempts, run_after, last_error
                      FROM jobs
                     WHERE message_id = m.id
                     ORDER BY id DESC
                     LIMIT 1
              ) j ON true
             WHERE NOT EXISTS (
                   SELECT 1 FROM classifications c
                    WHERE c.message_id = m.id
                      AND c.model_id = :model_id
                      AND c.prompt_version = :prompt_version
             )
             -- id DESC breaks ties: bulk mail routinely shares a timestamp
             -- to the second, and an unstable sort under OFFSET paging
             -- makes rows repeat on one page and vanish from the next.
             ORDER BY m.internal_date DESC, m.id DESC
             LIMIT :limit OFFSET :offset
            """
        ),
        {
            "model_id": settings.vllm_model_id,
            "prompt_version": settings.prompt_version,
            "limit": limit,
            "offset": offset,
        },
    )
    return [dict(r) for r in result.mappings()]


async def count_unprocessed(session: AsyncSession) -> int:
    settings = get_settings()
    n = await session.scalar(
        text(
            """
            SELECT count(*) FROM messages m
             WHERE NOT EXISTS (
                   SELECT 1 FROM classifications c
                    WHERE c.message_id = m.id
                      AND c.model_id = :model_id
                      AND c.prompt_version = :prompt_version
             )
            """
        ),
        {
            "model_id": settings.vllm_model_id,
            "prompt_version": settings.prompt_version,
        },
    )
    return int(n or 0)


async def clear_pending_jobs(session: AsyncSession) -> int:
    """Drop queued work. Never touches 'running' rows.

    Deleting a claimed job would orphan the worker holding it -- it would
    finish the pipeline and then update a row that no longer exists. Those are
    left alone; the stuck-job sweep deals with them if the worker dies.
    Messages are untouched, so anything cleared here can be re-queued from the
    same console.
    """
    result = await session.execute(
        text("DELETE FROM jobs WHERE status = 'pending' RETURNING id")
    )
    return len(result.all())


async def enqueue_unprocessed(session: AsyncSession, limit: int = 10_000) -> int:
    """Queue every unprocessed message that has no live job."""
    settings = get_settings()
    result = await session.execute(
        text(
            """
            SELECT m.id FROM messages m
             WHERE NOT EXISTS (
                   SELECT 1 FROM classifications c
                    WHERE c.message_id = m.id
                      AND c.model_id = :model_id
                      AND c.prompt_version = :prompt_version
             )
               AND NOT EXISTS (
                   SELECT 1 FROM jobs j
                    WHERE j.message_id = m.id AND j.status IN ('pending', 'running')
             )
             ORDER BY m.internal_date DESC, m.id DESC
             LIMIT :limit
            """
        ),
        {
            "model_id": settings.vllm_model_id,
            "prompt_version": settings.prompt_version,
            "limit": limit,
        },
    )
    ids = [r[0] for r in result]
    for message_id in ids:
        await queue.enqueue(session, message_id, notify=False)
    if ids:
        # One NOTIFY for the batch; workers drain the whole queue on a wake.
        await session.execute(text("SELECT pg_notify('triage_jobs', 'enqueue_all')"))
    return len(ids)


async def pending_job_message_ids(session: AsyncSession, limit: int = 100) -> list[int]:
    result = await session.execute(
        select(JobRow.message_id).where(JobRow.status == "pending").limit(limit)
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# sync state
# ---------------------------------------------------------------------------


async def get_sync_state(session: AsyncSession) -> SyncState:
    row = await session.get(SyncStateRow, 1)
    if row is None:
        row = SyncStateRow(id=1)
        session.add(row)
        await session.flush()
    return SyncState.model_validate(row)


async def update_sync_state(
    session: AsyncSession,
    *,
    history_id: str | None = None,
    watch_expiry: datetime | None = None,
    touch_reconcile: bool = False,
) -> SyncState:
    await get_sync_state(session)  # ensure the row exists
    values: dict[str, Any] = {}
    if history_id is not None:
        values["history_id"] = history_id
    if watch_expiry is not None:
        values["watch_expiry"] = watch_expiry
    if touch_reconcile:
        values["last_reconcile_at"] = datetime.now(UTC)
    if values:
        await session.execute(update(SyncStateRow).where(SyncStateRow.id == 1).values(**values))
    row = await session.get(SyncStateRow, 1)
    await session.refresh(row)
    return SyncState.model_validate(row)
