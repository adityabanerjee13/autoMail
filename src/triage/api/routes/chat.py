"""The chat console: a JSON API, plus the shell that loads the React app.

This is the one part of the app that is not server-rendered HTML. A chat
transcript is genuinely stateful in the browser -- a scroll position to hold, a
draft to preserve across refreshes, an autoscroll that must not fight the user
who has scrolled up to read something -- and expressing that in swapped HTML
fragments means fighting the framework rather than using it.

The rest of the app keeps its htmx forms and its works-without-JavaScript
guarantee. That guarantee is dropped *here only*, deliberately: the ops console
at /queue still drives every one of these operations without JavaScript, so
nothing is unreachable if the bundle fails to load.

``POST /api/chat/threads/{id}/messages`` returns immediately and the turn runs
as a detached task. A turn is 15-120 seconds on this hardware -- several
~6k-token prefills on an Arc iGPU, sometimes a three-minute IMAP sync -- so a
request that blocked for it would burn a worker, hit proxy timeouts, and leave
the user with a dead tab. The client polls while a turn is in flight.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from triage.agent.loop import run_turn
from triage.agent.tools import REGISTRY
from triage.api.deps import STATIC_DIR, get_session
from triage.api.tasks import agent_runner, runner
from triage.config import get_settings
from triage.db import chat
from triage.db.engine import sessionmaker_for
from triage.llm.client import LLMClient

router = APIRouter()
log = logging.getLogger("triage.api.chat")

SPA_INDEX = STATIC_DIR / "app" / "index.html"

#: Sent once with the thread payload so the client can render a confirmation
#: banner that states the real consequence, without duplicating the wording.
CONFIRM_PROMPTS = {
    name.value: spec.confirm_prompt for name, spec in REGISTRY.items() if spec.destructive
}


def _owner() -> str:
    return chat.owner_key(get_settings())


def _message_json(row) -> dict[str, Any]:
    return {
        "id": row.id,
        "seq": row.seq,
        "role": row.role,
        "content": row.content,
        "thought": row.thought,
        "tool_name": row.tool_name,
        "tool_args": row.tool_args,
        # Only the model-facing text is exposed; meta is internal bookkeeping.
        "tool_text": (row.tool_result or {}).get("text") if row.tool_result else None,
        "status": row.status,
        "error_code": row.error_code,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ---------------------------------------------------------------------------
# the shell
# ---------------------------------------------------------------------------


@router.get("/chat")
@router.get("/chat/{thread_id}")
async def chat_app(thread_id: int | None = None):
    """Serve the built React bundle. Routing past /chat happens client-side."""
    if not SPA_INDEX.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "The chat UI has not been built. Run `npm ci && npm run build` in "
                "frontend/, which writes into src/triage/api/static/app/."
            ),
        )
    # no-store: the shell names hashed asset files, and a cached shell would go
    # on pointing at a bundle that no longer exists after a rebuild.
    return FileResponse(SPA_INDEX, headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# threads
# ---------------------------------------------------------------------------


@router.get("/api/chat/threads")
async def list_threads(session: AsyncSession = Depends(get_session)):
    owner = _owner()
    return {
        "owner": owner,
        "threads": [
            {"id": t["id"], "title": t["title"], "updated_at": t["updated_at"].isoformat()}
            for t in await chat.list_threads(session, owner)
        ],
    }


@router.post("/api/chat/threads", status_code=201)
async def create_thread(session: AsyncSession = Depends(get_session)):
    thread_id = await chat.create_thread(session, _owner())
    await session.commit()
    return {"id": thread_id}


@router.get("/api/chat/threads/{thread_id}")
async def get_thread(thread_id: int, session: AsyncSession = Depends(get_session)):
    """The whole transcript plus the live state the client polls for."""
    thread = await chat.get_thread(session, thread_id, _owner())
    if thread is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    rows = await chat.messages_for_render(session, thread_id)
    return {
        "thread": {
            "id": thread["id"],
            "title": thread["title"],
            "updated_at": thread["updated_at"].isoformat(),
        },
        "messages": [_message_json(r) for r in rows],
        # Read at the moment of use: the task can finish between two awaits.
        "busy": agent_runner.is_running(thread_id),
        "busy_elsewhere": agent_runner.is_busy() and not agent_runner.is_running(thread_id),
        # Honest about why replies are slow while the mailbox is classifying.
        "queue_running": runner.running,
        "confirm_prompts": CONFIRM_PROMPTS,
    }


@router.delete("/api/chat/threads/{thread_id}")
async def delete_thread(thread_id: int, session: AsyncSession = Depends(get_session)):
    agent_runner.cancel(thread_id)
    deleted = await chat.delete_thread(session, thread_id, _owner())
    await session.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="no such conversation")
    return {"ok": True}


# ---------------------------------------------------------------------------
# turns
# ---------------------------------------------------------------------------


@router.post("/api/chat/threads/{thread_id}/messages", status_code=202)
async def post_message(
    thread_id: int,
    payload: dict = Body(...),
    session: AsyncSession = Depends(get_session),
):
    owner = _owner()
    thread = await chat.get_thread(session, thread_id, owner)
    if thread is None:
        raise HTTPException(status_code=404, detail="no such conversation")

    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="a message cannot be empty")

    # Checked before writing anything: a question written into a thread whose
    # previous one is still running would be ignored by that turn and look to
    # the user as though the agent had lost it.
    if agent_runner.is_busy():
        return {
            "started": False,
            "note": (
                "Still replying to the last message."
                if agent_runner.is_running(thread_id)
                else "Another conversation is still being answered. One at a time."
            ),
        }

    turn_id = await chat.add_message(session, thread_id, role="user", content=text)
    # The first message names the thread. An LLM-generated title would be ten
    # seconds of GPU for a sidebar label on this box.
    await chat.touch_thread(
        session, thread_id, title=text if thread["title"] == "New chat" else None
    )
    await session.commit()

    _start_turn(thread_id, turn_id, text, owner)
    return {"started": True, "turn_id": turn_id}


@router.post("/api/chat/threads/{thread_id}/confirm")
async def confirm(thread_id: int, payload: dict = Body(...)):
    """Answer a destructive tool's confirmation. The turn is blocked on it."""
    message_id = payload.get("message_id")
    if not isinstance(message_id, int):
        raise HTTPException(status_code=422, detail="message_id is required")
    resolved = agent_runner.resolve_confirmation(message_id, bool(payload.get("approved")))
    # A double click, a stale tab, or a confirmation stranded by a restart.
    # None of those deserves a 500.
    return {"resolved": resolved}


@router.post("/api/chat/threads/{thread_id}/stop")
async def stop(thread_id: int):
    return {"cancelled": agent_runner.cancel(thread_id)}


@router.post("/api/chat/threads/{thread_id}/retry")
async def retry(thread_id: int, session: AsyncSession = Depends(get_session)):
    """Re-run the last turn: keep the question, drop the failed answer."""
    owner = _owner()
    if await chat.get_thread(session, thread_id, owner) is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    if agent_runner.is_busy():
        return {"started": False, "note": "Something is already running."}

    turn_id = await chat.last_turn_id(session, thread_id)
    if turn_id is None:
        return {"started": False, "note": "Nothing to retry."}
    row = await chat.get_message(session, turn_id)
    await chat.delete_turn_tail(session, turn_id)
    await session.commit()

    _start_turn(thread_id, turn_id, row.content, owner)
    return {"started": True, "turn_id": turn_id}


def _start_turn(thread_id: int, turn_id: int, text: str, owner: str) -> None:
    """Hand the turn to a detached task.

    ``sessionmaker_for()``, never the request's session: this outlives the
    response by up to ten minutes, and a request-scoped session is closed the
    moment the response is sent. That is the single most likely bug here.
    """
    settings = get_settings()
    coro = run_turn(
        sessionmaker_for(),
        LLMClient(settings),
        thread_id=thread_id,
        turn_id=turn_id,
        user_message=text,
        owner=owner,
        confirm=agent_runner.wait_for_confirmation,
        settings=settings,
    )
    if not agent_runner.start(thread_id, coro):
        log.info("thread %s not started; the agent is busy", thread_id)
