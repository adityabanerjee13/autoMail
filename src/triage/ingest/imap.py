"""IMAP source, used for the initial historical import.

Why IMAP and not the API for bulk: IMAP is bandwidth-limited (~2.5 GB/day on
Gmail) rather than quota-unit-limited. At 20 units per messages.get the API
caps out around 300 bodies a minute; IMAP will move tens of thousands
overnight.

Two Gmail-specific traps this module handles:

* Labels are exposed as folders, so one message appears in INBOX, in
  [Gmail]/All Mail, and in every label folder. We sync [Gmail]/All Mail only
  and key on X-GM-MSGID, so a message is stored once no matter how many labels
  it carries.
* X-GM-MSGID in hex is the Gmail API's message id. Keying on it means the
  overnight IMAP import and the steady-state API sync deduplicate against each
  other for free.

Scope note: Gmail's IMAP XOAUTH2 mechanism wants the full
``https://mail.google.com/`` scope -- gmail.readonly is not documented to work
here. Either request that scope for the backfill run specifically (the sync
daemon keeps gmail.readonly), or set IMAP_PASSWORD to an app password. See the
README.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime

from imapclient import IMAPClient

from triage.config import Settings, get_settings
from triage.ingest.auth import imap_credentials, load_credentials
from triage.ingest.base import Checkpoint
from triage.parse.mime import ParsedEmail, gmail_id_from_msgid, parse_eml

log = logging.getLogger(__name__)

#: Gmail's IMAP extension keys. X-GM-EXT-1 is advertised on connect.
GM_MSGID = b"X-GM-MSGID"
GM_THRID = b"X-GM-THRID"
GM_LABELS = b"X-GM-LABELS"

#: Messages per FETCH. Larger batches are faster but a failure re-fetches the
#: whole chunk, and Gmail occasionally drops a connection mid-response.
CHUNK = 50


class ImapSource:
    """Implements MailSource over Gmail IMAP."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Three sources, most explicit first: constructor args, the credentials
        # the dashboard stored, then env. Absent a password entirely we fall
        # back to XOAUTH2 with the OAuth token.
        stored = imap_credentials(self.settings)
        self._username = (
            username
            or (stored[0] if stored else None)
            or self.settings.imap_username
            or None
        )
        self._password = (
            password
            or (stored[1] if stored else None)
            or os.environ.get("IMAP_PASSWORD")
        )
        self._pool: list[IMAPClient] = []

    # -- connections -------------------------------------------------------

    def _connect(self) -> IMAPClient:
        client = IMAPClient(self.settings.imap_host, port=self.settings.imap_port, ssl=True)
        # IMAPClient defaults normalise_times to True, which converts
        # INTERNALDATE to a *naive datetime in the server's local timezone*.
        # Tagging that as UTC -- which is the obvious thing to do with a naive
        # value -- shifts every timestamp by the local offset, and then the UI
        # converting UTC to local shifts it again. On a +05:30 box that put mail
        # 11 hours into the future, and dated anything after 18:30 tomorrow.
        # Off means INTERNALDATE arrives as an aware UTC datetime, which is what
        # the column actually wants.
        client.normalise_times = False
        username = self._username
        if not username:
            raise ValueError(
                "no IMAP username: connect an account in the console, or set IMAP_USERNAME"
            )
        if self._password:
            client.login(username, self._password)
        else:
            creds = load_credentials(self.settings)
            client.oauth2_login(username, creds.token)
        # readonly: this process must never mutate the mailbox.
        client.select_folder(self.settings.imap_folder, readonly=True)
        return client

    async def _acquire(self) -> IMAPClient:
        if self._pool:
            return self._pool.pop()
        return await asyncio.to_thread(self._connect)

    def _release(self, client: IMAPClient) -> None:
        self._pool.append(client)

    async def close(self) -> None:
        pool, self._pool = self._pool, []
        for client in pool:
            try:
                await asyncio.to_thread(client.logout)
            except Exception:  # noqa: BLE001 - closing must not raise
                pass

    # -- search ------------------------------------------------------------

    async def search(self, criteria: list) -> list[int]:
        client = await self._acquire()
        try:
            uids = await asyncio.to_thread(client.search, criteria)
            return list(uids)
        finally:
            self._release(client)

    # -- fetch -------------------------------------------------------------

    def _fetch_chunk_sync(self, client: IMAPClient, uids: list[int]) -> list[ParsedEmail]:
        # BODY.PEEK[] rather than RFC822: RFC822 sets the \Seen flag, and this
        # is meant to be a read-only pass over the archive.
        raw = client.fetch(uids, [b"BODY.PEEK[]", GM_MSGID, GM_THRID, GM_LABELS, b"INTERNALDATE"])
        out: list[ParsedEmail] = []
        for uid, data in raw.items():
            body = data.get(b"BODY[]") or data.get(b"RFC822")
            msgid = data.get(GM_MSGID)
            if not body or not msgid:
                log.warning("uid %s returned no body or no X-GM-MSGID; skipping", uid)
                continue
            thrid = data.get(GM_THRID) or msgid
            labels = [_decode_label(x) for x in (data.get(GM_LABELS) or ())]
            internal = data.get(b"INTERNALDATE")
            if isinstance(internal, datetime) and internal.tzinfo is None:
                internal = internal.replace(tzinfo=UTC)
            out.append(
                parse_eml(
                    body,
                    gmail_id=gmail_id_from_msgid(msgid),
                    thread_id=gmail_id_from_msgid(thrid),
                    labels=labels,
                    internal_date=internal if isinstance(internal, datetime) else None,
                )
            )
        return out

    async def fetch_uids(self, uids: Iterable[int]) -> AsyncIterator[ParsedEmail]:
        """Fetch and parse UIDs across a pool of connections."""
        chunks = _chunked(list(uids), CHUNK)
        semaphore = asyncio.Semaphore(self.settings.imap_concurrency)

        async def one(chunk: list[int]) -> list[ParsedEmail]:
            async with semaphore:
                client = await self._acquire()
                try:
                    result = await asyncio.to_thread(self._fetch_chunk_sync, client, chunk)
                except Exception as exc:  # noqa: BLE001
                    # A dropped connection poisons the client; do not return it
                    # to the pool.
                    log.warning("chunk of %d failed (%s); dropping connection", len(chunk), exc)
                    try:
                        await asyncio.to_thread(client.logout)
                    except Exception:  # noqa: BLE001
                        pass
                    return []
                self._release(client)
                return result

        # Bounded windows rather than one gather over everything: the whole
        # mailbox worth of ParsedEmail objects will not fit in 18 GB.
        window = self.settings.imap_concurrency * 2
        for i in range(0, len(chunks), window):
            batch = chunks[i : i + window]
            for result in await asyncio.gather(*(one(c) for c in batch)):
                for parsed in result:
                    yield parsed

    # -- MailSource --------------------------------------------------------

    async def backfill(self, before: datetime) -> AsyncIterator[ParsedEmail]:
        uids = await self.search(["BEFORE", before.date()])
        log.info("imap backfill: %d messages before %s", len(uids), before.date())
        # Newest first: the recent archive is the part worth having early if
        # the run is interrupted.
        async for parsed in self.fetch_uids(sorted(uids, reverse=True)):
            yield parsed

    async def fetch_since(self, checkpoint: Checkpoint) -> AsyncIterator[ParsedEmail]:
        """Present to satisfy MailSource; the steady-state path is the API.

        IMAP UID tracking is deliberately not implemented -- the handoff calls
        for a single watermark mechanism (historyId) and two would drift.
        """
        if checkpoint.internal_date is None:
            raise ValueError("ImapSource.fetch_since needs an internal_date checkpoint")
        uids = await self.search(["SINCE", checkpoint.internal_date.date()])
        async for parsed in self.fetch_uids(sorted(uids)):
            yield parsed


def _decode_label(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip('"')
    return str(value).strip('"')


def _chunked(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def sync_recent(
    days: int = 7, *, limit: int = 500, settings: Settings | None = None
) -> dict:
    """Pull the last `days` of mail over IMAP and queue it.

    The console's sync button when the account was connected with an App
    Password. This is not the steady-state path -- that is historyId over the
    API, and it stays that way, because IMAP has no equivalent watermark and
    two watermark mechanisms would drift. What this is: a bounded catch-up that
    is idempotent, because ON CONFLICT DO NOTHING on gmail_id absorbs the
    overlap every run produces.
    """
    from datetime import datetime, timedelta

    from triage.db import repo
    from triage.db.engine import sessionmaker_for
    from triage.ingest.base import to_message_in

    settings = settings or get_settings()
    sessions = sessionmaker_for()
    source = ImapSource(settings)
    since = (datetime.now(UTC) - timedelta(days=days)).date()

    stored = duplicates = failed = 0
    try:
        uids = await source.search(["SINCE", since])
        log.info("imap sync: %d message(s) since %s", len(uids), since)
        # Newest first, so a capped run gets the mail that matters most.
        async for parsed in source.fetch_uids(sorted(uids, reverse=True)[:limit]):
            try:
                async with sessions() as session:
                    message, _ = await repo.ingest_message(session, to_message_in(parsed))
                    await session.commit()
                if message is None:
                    duplicates += 1
                else:
                    stored += 1
            except Exception:  # noqa: BLE001 - one bad message must not end the sync
                failed += 1
                log.exception("failed to store gmail_id=%s", parsed.gmail_id)
    finally:
        await source.close()

    log.info("imap sync done: stored=%d dup=%d failed=%d", stored, duplicates, failed)
    return {"stored": stored, "duplicates": duplicates, "failed": failed, "scanned": len(uids)}
