"""Browsing: the message list and one message's full judgment history."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from triage.api.deps import get_session, templates
from triage.db import repo
from triage.db.models import ClassificationRow
from triage.taxonomy import category_values

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    page: int = 1,
    category: str | None = None,
    important: str | None = None,
    q: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    limit = 50
    # Parsed by hand rather than declared as `bool | None`: the filter form
    # submits important="" for "any", which a bool parameter rejects with a 422.
    important_filter = {"true": True, "false": False}.get((important or "").lower())
    rows = await repo.list_messages(
        session,
        limit=limit,
        offset=(page - 1) * limit,
        category=category or None,
        important=important_filter,
        search=q or None,
    )
    stats = await repo.stats(session)
    template = "_rows.html" if request.headers.get("HX-Request") else "messages.html"
    return templates.TemplateResponse(
        request,
        template,
        {
            "rows": rows,
            "stats": stats,
            "categories": category_values(),
            "filters": {"category": category, "important": important_filter, "q": q},
            "page": page,
            "has_next": len(rows) == limit,
        },
    )


@router.get("/messages/{message_id}", response_class=HTMLResponse)
async def detail(
    request: Request,
    message_id: int,
    session: AsyncSession = Depends(get_session),
):
    message = await repo.get_message(session, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="no such message")

    # The full append-only history, newest first. Seeing an LLM row and the
    # human row that superseded it side by side is the whole reason the table
    # is append-only.
    history = (
        (
            await session.execute(
                select(ClassificationRow)
                .where(ClassificationRow.message_id == message_id)
                .order_by(desc(ClassificationRow.id))
            )
        )
        .scalars()
        .all()
    )
    return templates.TemplateResponse(
        request,
        "message_detail.html",
        {
            "message": message,
            "history": history,
            "categories": category_values(),
        },
    )
