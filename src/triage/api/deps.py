"""Shared API dependencies: the template environment and the session provider.

Separate from ``app.py`` to break the import cycle -- ``app.create_app()``
imports the route modules, so the route modules cannot import ``app``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from triage.db.engine import sessionmaker_for

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def localdt(value: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Render a stored timestamp in this machine's timezone.

    Every timestamp in the database is timestamptz in UTC, which is right for
    storage and wrong for a screen: calling strftime on it directly shows a
    time that is hours off from the one Gmail displays, and the mismatch is
    silent. ``astimezone()`` with no argument converts to local time.
    """
    if value is None:
        return ""
    return value.astimezone().strftime(fmt)


def localdt_full(value: datetime | None) -> str:
    """Full timestamp with zone, for tooltips."""
    if value is None:
        return ""
    return value.astimezone().strftime("%A %d %B %Y, %H:%M:%S %Z")


templates.env.filters["localdt"] = localdt
templates.env.filters["localdt_full"] = localdt_full


async def get_session() -> AsyncIterator[AsyncSession]:
    """One session per request. Routes that write commit explicitly."""
    async with sessionmaker_for()() as session:
        yield session
