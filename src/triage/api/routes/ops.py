"""The operations console: connect Gmail, sync, and drive the queue.

Every mutation here is a POST that returns the refreshed console fragment, so
the page works with htmx and works identically as a plain form post without it.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from triage.api.deps import get_session, templates
from triage.api.tasks import gmail_connect, runner
from triage.config import get_settings
from triage.db import queue, repo
from triage.ingest import auth

router = APIRouter()
log = logging.getLogger("triage.api.ops")

PAGE_SIZE = 50


async def _console(session: AsyncSession) -> dict:
    """Everything the console renders. One place, so every action can reuse it."""
    settings = get_settings()
    gmail = auth.status(settings)
    imap_user = (auth.imap_credentials(settings) or [None])[0]
    return {
        "jobs": await queue.counts(session),
        # Connected but empty is the state that confuses people: signing in
        # stores a credential, it does not fetch anything. The console says so
        # explicitly rather than showing an empty table.
        "messages_total": await session.scalar(text("SELECT count(*) FROM messages")),
        "unprocessed_count": await repo.count_unprocessed(session),
        "rows": await repo.unprocessed(session, limit=PAGE_SIZE),
        "runner": runner.status(),
        "gmail": {
            # Connected means "we can read the mailbox", by either route.
            # Reporting only the OAuth token here told anyone who signed in with
            # an App Password that they were not connected, directly under a
            # panel naming their address.
            "connected": gmail.connected or bool(imap_user),
            "method": "oauth" if gmail.connected else ("imap" if imap_user else None),
            "oauth_connected": gmail.connected,
            "reason": gmail.reason,
            "email": auth.account_email(settings),
            "connecting": gmail_connect.in_progress,
            "error": gmail_connect.error,
            "last_sync": getattr(gmail_connect, "last_sync", None),
            # Until this exists there is nothing to consent against, so the
            # console shows the setup steps instead of a dead button.
            "has_client_secrets": auth.has_client_secrets(settings),
            "credentials_path": str(settings.resolve(settings.gmail_credentials_path)),
            "auth_url": gmail_connect.auth_url,
            "scope": auth.SCOPES[0],
            # The App Password route -- OpenClaw's method, and the one that
            # needs no Google Cloud project.
            "imap": imap_user,
        },
        "sync": await repo.get_sync_state(session),
        "versions": {
            "model_id": settings.vllm_model_id,
            "prompt_version": settings.prompt_version,
        },
    }


@router.get("/queue", response_class=HTMLResponse)
async def console(request: Request, session: AsyncSession = Depends(get_session)):
    ctx = await _console(session)
    # htmx polls this same route for the live panel; a full page load gets the
    # whole document.
    template = "_console.html" if request.headers.get("HX-Request") else "queue.html"
    return templates.TemplateResponse(request, template, ctx)


# ---------------------------------------------------------------------------
# gmail
# ---------------------------------------------------------------------------


@router.post("/gmail/connect", response_class=HTMLResponse)
async def gmail_login(request: Request, session: AsyncSession = Depends(get_session)):
    """Start OAuth consent. A browser opens on the machine running the API."""
    settings = get_settings()
    creds_path = settings.resolve(settings.gmail_credentials_path)
    if not creds_path.exists():
        # Far more useful than letting the flow fail three seconds later with a
        # FileNotFoundError nobody sees.
        gmail_connect.error = (
            f"No OAuth client secrets at {creds_path}. Create a Desktop app client in "
            "Google Cloud Console and save the download there."
        )
    else:
        gmail_connect.start()
    return templates.TemplateResponse(request, "_console.html", await _console(session))



@router.post("/gmail/credentials", response_class=HTMLResponse)
async def upload_client_secrets(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Accept the OAuth client secrets JSON downloaded from Google Cloud Console.

    Uploading beats "copy this file to data/secrets/": it is the step people
    get wrong, and the validation here names the mistake instead of letting it
    surface later as an opaque failure inside the consent flow.
    """
    raw = await file.read()
    try:
        path = auth.save_client_secrets(raw, get_settings())
        gmail_connect.error = None
        log.info("client secrets uploaded to %s", path)
    except ValueError as exc:
        gmail_connect.error = f"{file.filename}: {exc}"
    return templates.TemplateResponse(request, "_console.html", await _console(session))



@router.post("/gmail/app-password", response_class=HTMLResponse)
async def connect_app_password(
    request: Request,
    email: str = Form(...),
    app_password: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    """Connect over IMAP with a Google App Password -- no Google Cloud project.

    Verified against the live server before being stored. Saving a credential
    that does not work just moves the failure into the next sync, where it
    looks like a sync bug rather than a typo.
    """
    settings = get_settings()
    try:
        # Format first, network second: no point asking Google about a string
        # that cannot be an App Password.
        email, app_password = auth.normalize_imap_credentials(email, app_password)
        detail = await asyncio.to_thread(auth.verify_imap, email, app_password, settings)
        auth.save_imap_credentials(email, app_password, settings)
        gmail_connect.error = None
        log.info("connected %s over IMAP: %s", email, detail)
    except ValueError as exc:
        gmail_connect.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - imaplib raises bare exceptions
        text = str(exc)
        if "AUTHENTICATIONFAILED" in text.upper() or "Invalid credentials" in text:
            gmail_connect.error = (
                "Google rejected those credentials. Two usual causes: this is your "
                "normal password rather than a 16-character App Password, or IMAP "
                "is disabled in Gmail settings (Forwarding and POP/IMAP)."
            )
        else:
            gmail_connect.error = f"IMAP login failed: {text}"
        log.warning("imap connect failed for %s: %s", email, text)
    return templates.TemplateResponse(request, "_console.html", await _console(session))


@router.post("/gmail/app-password/forget", response_class=HTMLResponse)
async def forget_app_password(
    request: Request, session: AsyncSession = Depends(get_session)
):
    auth.forget_imap_credentials(get_settings())
    log.info("imap credentials removed at operator request")
    return templates.TemplateResponse(request, "_console.html", await _console(session))


@router.post("/gmail/imap-sync", response_class=HTMLResponse)
async def imap_sync(
    request: Request,
    days: int = Form(7),
    session: AsyncSession = Depends(get_session),
):
    """Pull recent mail over IMAP and queue it."""
    from triage.ingest.imap import sync_recent

    try:
        result = await sync_recent(days=days, settings=get_settings())
        gmail_connect.error = None
        gmail_connect.last_sync = (
            f"IMAP sync: {result['stored']} new, {result['duplicates']} already stored"
            f" (scanned {result['scanned']} from the last {days} days)"
        )
    except Exception as exc:  # noqa: BLE001 - reported in the console
        log.exception("imap sync failed")
        gmail_connect.error = f"IMAP sync failed: {exc}"
    return templates.TemplateResponse(request, "_console.html", await _console(session))


@router.post("/gmail/disconnect", response_class=HTMLResponse)
async def gmail_logout(request: Request, session: AsyncSession = Depends(get_session)):
    settings = get_settings()
    token = settings.resolve(settings.gmail_token_path)
    token.unlink(missing_ok=True)
    # The client secrets stay. Disconnecting an account is not the same as
    # throwing away the OAuth client, and re-uploading it every time would be
    # tedious.
    log.info("gmail token removed at operator request")
    return templates.TemplateResponse(request, "_console.html", await _console(session))


@router.post("/gmail/sync", response_class=HTMLResponse)
async def gmail_sync(request: Request, session: AsyncSession = Depends(get_session)):
    """Pull new mail now, rather than waiting for Pub/Sub or the hourly sweep."""
    # The shared daemon, not a fresh one: its lock only serialises callers that
    # hold the same instance. See the comment on the singleton in tasks.py.
    from triage.api.tasks import sync_daemon

    try:
        result = await sync_daemon.sync_once()
        log.info("manual sync: %s", result)
    except Exception as exc:  # noqa: BLE001 - reported in the console
        log.exception("manual sync failed")
        gmail_connect.error = f"Sync failed: {exc}"
    ctx = await _console(session)
    return templates.TemplateResponse(request, "_console.html", ctx)


# ---------------------------------------------------------------------------
# queue control
# ---------------------------------------------------------------------------


@router.post("/queue/run", response_class=HTMLResponse)
async def run_queue(request: Request, session: AsyncSession = Depends(get_session)):
    """Queue everything unprocessed, then drain it in this process."""
    n = await repo.enqueue_unprocessed(session)
    await session.commit()
    started = runner.start()
    log.info("run queue: enqueued=%d runner_started=%s", n, started)
    return templates.TemplateResponse(request, "_console.html", await _console(session))


@router.post("/queue/stop", response_class=HTMLResponse)
async def stop_queue(request: Request, session: AsyncSession = Depends(get_session)):
    """Stop after the message currently in flight -- never mid-message."""
    runner.stop()
    return templates.TemplateResponse(request, "_console.html", await _console(session))


@router.post("/queue/clear", response_class=HTMLResponse)
async def clear_queue(request: Request, session: AsyncSession = Depends(get_session)):
    """Delete pending jobs. Running jobs and every message are left alone."""
    n = await repo.clear_pending_jobs(session)
    await session.commit()
    log.warning("cleared %d pending job(s) at operator request", n)
    return templates.TemplateResponse(request, "_console.html", await _console(session))


@router.post("/queue/retry-dead", response_class=HTMLResponse)
async def retry_dead(request: Request, session: AsyncSession = Depends(get_session)):
    n = await queue.retry_dead(session)
    await session.commit()
    log.info("returned %d dead-lettered job(s) to the queue", n)
    return templates.TemplateResponse(request, "_console.html", await _console(session))



@router.post("/messages/{message_id}/dequeue", response_class=HTMLResponse)
async def dequeue_one(
    request: Request, message_id: int, session: AsyncSession = Depends(get_session)
):
    """Take one message back out of the queue, without deleting the message.

    The counterpart to Clear queue, which is all-or-nothing. Skipping a single
    piece of mail should not mean flushing everything else you queued.
    """
    n = await queue.remove_pending_for(session, message_id)
    await session.commit()
    log.info("removed %d queued job(s) for message %s", n, message_id)
    return templates.TemplateResponse(request, "_console.html", await _console(session))


@router.post("/messages/{message_id}/process", response_class=HTMLResponse)
async def process_one(
    request: Request, message_id: int, session: AsyncSession = Depends(get_session)
):
    """Classify exactly this message -- not the rest of the queue.

    It queues a job and then claims that specific job id, so the button does
    what it says. Calling the general drain here used to classify the whole
    mailbox off one click.
    """
    message = await repo.get_message(session, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="no such message")
    # Reuse the queued job if this message already has one, rather than
    # stacking a second one on top of it.
    job_id = await queue.pending_job_for(session, message_id)
    if job_id is None:
        job_id = await queue.enqueue(session, message_id)
    await session.commit()
    if not runner.start_one(job_id):
        # Something is already running; the job stays queued and gets picked
        # up rather than being silently dropped.
        log.info("message %s queued; runner busy", message_id)
    return templates.TemplateResponse(request, "_console.html", await _console(session))
