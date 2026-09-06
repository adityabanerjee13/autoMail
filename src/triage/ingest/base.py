"""The ingest boundary.

Both sources implement this protocol, and nothing above ``ingest/`` knows which
one is running. That is what lets the IMAP backfill and the Gmail API steady
state write to the same table without the pipeline noticing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from triage.parse.mime import ParsedEmail
from triage.schemas import MessageIn


@dataclass(slots=True)
class Checkpoint:
    """Where a source resumes from.

    ``history_id`` is Gmail's watermark; ``internal_date`` is the fallback for
    sources without one (and for the bounded resync after a 404).
    """

    history_id: str | None = None
    internal_date: datetime | None = None


class HistoryTooOldError(RuntimeError):
    """history.list returned 404: the watermark is outside Gmail's ~7 day
    retention. The caller must fall back to a bounded messages.list resync and
    re-checkpoint."""


@runtime_checkable
class MailSource(Protocol):
    async def fetch_since(self, checkpoint: Checkpoint) -> AsyncIterator[ParsedEmail]:
        """Yield messages that arrived after the checkpoint."""
        ...

    async def backfill(self, before: datetime) -> AsyncIterator[ParsedEmail]:
        """Yield historical messages older than ``before``, newest first."""
        ...

    async def close(self) -> None:
        ...


def to_message_in(parsed: ParsedEmail) -> MessageIn:
    """The only conversion from parser output to a storable row.

    Both sources funnel through here so that a field added to ParsedEmail is
    either persisted for both or for neither.
    """
    headers = dict(parsed.headers)
    if parsed.has_attachments:
        headers["x-triage-has-attachments"] = "1"
    return MessageIn(
        gmail_id=parsed.gmail_id,
        thread_id=parsed.thread_id,
        message_id_hdr=parsed.message_id_hdr,
        from_addr=parsed.from_addr,
        to_addrs=parsed.to_addrs,
        cc_addrs=parsed.cc_addrs,
        subject=parsed.subject,
        body_clean=parsed.body_clean,
        snippet=parsed.snippet,
        labels=parsed.labels,
        headers=headers,
        internal_date=parsed.internal_date,
    )
