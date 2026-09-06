"""FastAPI app: read the judgments, correct the wrong ones.

The correction path is the point. Every human correction is an append to
``classifications`` with source='human', which is simultaneously the fix, the
audit trail, and the training data for phase 2.

Binds to 127.0.0.1 by default. This holds a personal mailbox and has no
authentication of its own.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from triage.api.deps import STATIC_DIR
from triage.api.routes import chat, health, messages, ops, review
from triage.config import get_settings

log = logging.getLogger("triage.api")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="email triage", version="0.1.0")

    # Vendored assets only -- this box makes no outbound requests except the
    # authenticated Gmail API calls. See base.html for how to fetch htmx.
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(ops.router)
    app.include_router(messages.router)
    app.include_router(review.router)
    app.include_router(health.router)
    app.include_router(chat.router)

    @app.on_event("startup")
    async def _log_config() -> None:  # pragma: no cover - startup noise
        log.info(
            "api up: model=%s prompt=%s schema=%s fewshot=%s",
            settings.vllm_model_id,
            settings.prompt_version,
            settings.schema_version,
            settings.enable_fewshot,
        )

    @app.on_event("startup")
    async def _sweep_chat() -> None:  # pragma: no cover - startup housekeeping
        """Fail chat rows a previous process left in flight.

        A turn is a task in this process and a confirmation is a future in this
        process, so a restart strands every 'running' and 'awaiting_confirm'
        row: nothing will ever resolve them, and the page would poll them
        forever. Same job, and the same reasoning, as queue.requeue_stuck.
        """
        from triage.agent.tools import sweep_interrupted
        from triage.db.engine import sessionmaker_for

        try:
            async with sessionmaker_for()() as session:
                n = await sweep_interrupted(session)
                await session.commit()
            if n:
                log.warning("swept %d chat row(s) interrupted by a restart", n)
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            log.warning("could not sweep interrupted chat rows: %s", exc)

    return app


app = create_app()


def main() -> None:  # pragma: no cover
    import uvicorn

    settings = get_settings()
    config = uvicorn.Config(
        "triage.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        # uvicorn builds its own proactor loop and passes it as an explicit
        # loop factory, so setting the event loop policy has no effect on it.
        # psycopg refuses to run async on a proactor loop, so drive uvicorn's
        # server inside a selector loop we own instead. Linux takes the normal
        # path; this branch exists only so the UI runs on a Windows dev box.
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
        return

    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
