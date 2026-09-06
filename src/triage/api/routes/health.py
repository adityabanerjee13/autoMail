"""Health and status.

Includes the Gmail credential state, because the failure this design most
expects -- a refresh token invalidated by a password change -- is silent
everywhere else. ``/health`` going amber here is what makes the reconnect
banner appear before a week of mail has gone missing.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from triage.api.deps import get_session, templates
from triage.config import get_settings
from triage.db import queue, repo
from triage.ingest import auth
from triage.llm.client import LLMClient

router = APIRouter()


@router.get("/health")
async def health(session: AsyncSession = Depends(get_session)) -> dict:
    settings = get_settings()

    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
        db_error = None
    except Exception as exc:  # noqa: BLE001 - health must not raise
        db_ok, db_error = False, repr(exc)

    gmail = auth.status(settings)
    llm = await LLMClient(settings).health()

    jobs: dict[str, int] = {}
    state = None
    if db_ok:
        jobs = await queue.counts(session)
        state = await repo.get_sync_state(session)

    return {
        "ok": db_ok and gmail.connected and llm.get("ok", False),
        "database": {"ok": db_ok, "error": db_error},
        "gmail": {
            "connected": gmail.connected,
            "reason": gmail.reason,
            # Drives the "reconnect account" path in the UI.
            "needs_reconnect": not gmail.connected,
        },
        "llm": llm,
        "jobs": jobs,
        "sync": state.model_dump(mode="json") if state else None,
        "versions": {
            "model_id": settings.vllm_model_id,
            "prompt_version": settings.prompt_version,
            "schema_version": settings.schema_version,
        },
    }


@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request, session: AsyncSession = Depends(get_session)):
    data = await health(session)
    stats = await repo.stats(session)
    return templates.TemplateResponse(
        request, "status.html", {"health": data, "stats": stats}
    )
