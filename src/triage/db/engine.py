"""Async engine and session plumbing.

One engine per process. The LISTEN connection in ``db/queue.py`` deliberately
does *not* come from this pool -- it is held open for the lifetime of the
worker and would otherwise starve the pool of a connection it can never
recycle.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from triage.config import settings


@lru_cache
def get_engine(url: str | None = None) -> AsyncEngine:
    return create_async_engine(
        url or settings.database_url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        echo=False,
    )


@lru_cache
def sessionmaker_for(url: str | None = None) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(url), expire_on_commit=False)


@asynccontextmanager
async def session_scope(url: str | None = None) -> AsyncIterator[AsyncSession]:
    """One transaction. Commits on clean exit, rolls back on exception.

    Message insert + job enqueue must share one of these -- that transactional
    guarantee is the whole reason there is no Redis in this design.
    """
    async with sessionmaker_for(url)() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def raw_dsn(url: str | None = None) -> str:
    """SQLAlchemy URL -> libpq DSN, for direct psycopg connections."""
    return (url or settings.database_url).replace("postgresql+psycopg://", "postgresql://")


def configure_event_loop_policy() -> None:
    """Make psycopg3's async mode usable on a Windows dev box.

    psycopg refuses to run on the ProactorEventLoop, which is Python's default
    on Windows. Production is Linux under systemd where this is a no-op, but
    without it nothing that touches the database runs locally. Called from
    every entry point before asyncio.run.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
