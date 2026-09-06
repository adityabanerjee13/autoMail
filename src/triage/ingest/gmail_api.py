"""Gmail API source: incremental sync over users.history.list.

Quota, per the 1 May 2026 change (verify before sizing anything bulk):

    1,200,000 units/min per project
        6,000 units/min per user per project   <- the binding limit here
   80,000,000 units/day per project billing threshold

    history.list 2 | messages.list 5 | messages.get 20 | threads.get 40

6,000 units/min against messages.get at 20 units caps this source at ~300
bodies per minute. ``fields=`` partial responses and HTTP batching cut payload
and round trips but *not* quota cost, which is why the historical import runs
over IMAP instead (see backfill.py).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from triage.config import Settings, get_settings
from triage.ingest.auth import load_credentials
from triage.ingest.base import Checkpoint, HistoryTooOldError
from triage.parse.mime import ParsedEmail, parse_gmail_message

log = logging.getLogger(__name__)

# Per-method quota cost, from the table above.
COST_HISTORY_LIST = 2
COST_MESSAGES_LIST = 5
COST_MESSAGES_GET = 20

USER_UNITS_PER_MINUTE = 6000

#: Enough to reconstruct a ParsedEmail and nothing more. Saves bandwidth and
#: parse time; does not save quota.
MESSAGE_FIELDS = "id,threadId,labelIds,internalDate,snippet,sizeEstimate,payload"


class QuotaLimiter:
    """Token bucket over Gmail quota units.

    Sized at 90% of the documented per-user ceiling: the server's accounting
    window does not line up with ours, and a 429 costs more than the headroom.
    """

    def __init__(self, units_per_minute: int = USER_UNITS_PER_MINUTE, safety: float = 0.9) -> None:
        self.rate = (units_per_minute * safety) / 60.0  # units per second
        self.capacity = units_per_minute * safety
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self, units: int) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= units:
                    self._tokens -= units
                    return
                await asyncio.sleep((units - self._tokens) / self.rate)


class GmailSource:
    """Implements MailSource over the Gmail REST API."""

    def __init__(
        self, settings: Settings | None = None, limiter: QuotaLimiter | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.limiter = limiter or QuotaLimiter()
        self._service: Any = None
        self.last_history_id: str | None = None

    # -- plumbing ----------------------------------------------------------

    async def service(self) -> Any:
        if self._service is None:
            creds = await asyncio.to_thread(load_credentials, self.settings)
            self._service = await asyncio.to_thread(
                build, "gmail", "v1", credentials=creds, cache_discovery=False
            )
        return self._service

    async def close(self) -> None:
        if self._service is not None:
            await asyncio.to_thread(self._service.close)
            self._service = None

    async def _execute(self, request, *, cost: int, retries: int = 5) -> Any:
        """Run one API call under the quota budget, retrying rate limits.

        404 is re-raised untouched -- on history.list it is meaningful (stale
        watermark) and must not be swallowed as a transient error.
        """
        for attempt in range(retries):
            await self.limiter.take(cost)
            try:
                return await asyncio.to_thread(request.execute)
            except HttpError as exc:
                code = exc.resp.status
                if code == 404 or (code < 500 and code != 429):
                    raise
                backoff = min(2**attempt, 32)
                log.warning("gmail %s; retrying in %ss", code, backoff)
                await asyncio.sleep(backoff)
        raise RuntimeError("gmail request failed after retries")

    # -- watermark ---------------------------------------------------------

    async def current_history_id(self) -> str:
        svc = await self.service()
        profile = await self._execute(
            svc.users().getProfile(userId=self.settings.gmail_user_id), cost=1
        )
        return str(profile["historyId"])

    # -- incremental sync --------------------------------------------------

    async def fetch_since(self, checkpoint: Checkpoint) -> AsyncIterator[ParsedEmail]:
        """Messages added since the checkpoint's historyId.

        Raises HistoryTooOldError on 404 so the daemon can fall back to a
        bounded messages.list resync -- Google keeps roughly one week of
        history and a daemon that was down over a holiday will land outside it.
        """
        if not checkpoint.history_id:
            raise HistoryTooOldError("no historyId watermark stored")

        svc = await self.service()
        message_ids: list[str] = []
        page_token: str | None = None
        newest = checkpoint.history_id

        while True:
            request = svc.users().history().list(
                userId=self.settings.gmail_user_id,
                startHistoryId=checkpoint.history_id,
                historyTypes=["messageAdded"],
                pageToken=page_token,
            )
            try:
                response = await self._execute(request, cost=COST_HISTORY_LIST)
            except HttpError as exc:
                if exc.resp.status == 404:
                    raise HistoryTooOldError(
                        f"historyId {checkpoint.history_id} is outside Gmail's retention"
                    ) from exc
                raise

            newest = str(response.get("historyId", newest))
            for record in response.get("history", []) or []:
                for added in record.get("messagesAdded", []) or []:
                    msg = added.get("message", {})
                    if msg.get("id"):
                        message_ids.append(msg["id"])

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        # history.list can report the same message more than once across
        # records; the DB would reject the duplicate anyway, but not fetching
        # it twice saves 20 units each time.
        for gmail_id in dict.fromkeys(message_ids):
            parsed = await self.get_message(gmail_id)
            if parsed is not None:
                yield parsed

        # Only advance the watermark once every message it covers has been
        # yielded. Advancing earlier would drop messages on a crash mid-loop.
        self.last_history_id = newest

    async def get_message(self, gmail_id: str) -> ParsedEmail | None:
        svc = await self.service()
        request = svc.users().messages().get(
            userId=self.settings.gmail_user_id,
            id=gmail_id,
            format="full",
            fields=MESSAGE_FIELDS,
        )
        try:
            raw = await self._execute(request, cost=COST_MESSAGES_GET)
        except HttpError as exc:
            if exc.resp.status == 404:
                # Deleted between the history record and the fetch.
                log.info("message %s no longer exists", gmail_id)
                return None
            raise
        return parse_gmail_message(raw)

    # -- bounded resync ----------------------------------------------------

    async def resync(
        self, since: datetime, *, max_messages: int = 2000
    ) -> AsyncIterator[ParsedEmail]:
        """messages.list fallback after a stale watermark.

        Bounded on purpose: this path exists to recover a few days, not to
        re-read the mailbox. ON CONFLICT DO NOTHING on gmail_id makes the
        overlap with already-stored messages free.
        """
        svc = await self.service()
        query = f"after:{int(since.timestamp())}"
        page_token: str | None = None
        seen = 0

        while seen < max_messages:
            request = svc.users().messages().list(
                userId=self.settings.gmail_user_id,
                q=query,
                pageToken=page_token,
                maxResults=min(500, max_messages - seen),
                fields="messages/id,nextPageToken",
            )
            response = await self._execute(request, cost=COST_MESSAGES_LIST)
            for stub in response.get("messages", []) or []:
                parsed = await self.get_message(stub["id"])
                seen += 1
                if parsed is not None:
                    yield parsed
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        self.last_history_id = await self.current_history_id()

    async def backfill(self, before: datetime) -> AsyncIterator[ParsedEmail]:
        """Present to satisfy MailSource. Use the IMAP source for real volume.

        At 20 units per body this path costs ~300 messages/minute against the
        per-user quota; IMAP is bandwidth-limited instead and is far cheaper
        for a full historical import.
        """
        log.warning("Gmail API backfill is quota-expensive; prefer ingest.imap.ImapSource")
        svc = await self.service()
        query = f"before:{before.strftime('%Y/%m/%d')}"
        page_token: str | None = None
        while True:
            request = svc.users().messages().list(
                userId=self.settings.gmail_user_id,
                q=query,
                pageToken=page_token,
                maxResults=500,
                fields="messages/id,nextPageToken",
            )
            response = await self._execute(request, cost=COST_MESSAGES_LIST)
            for stub in response.get("messages", []) or []:
                parsed = await self.get_message(stub["id"])
                if parsed is not None:
                    yield parsed
            page_token = response.get("nextPageToken")
            if not page_token:
                return

    # -- push notifications ------------------------------------------------

    async def watch(self) -> dict[str, Any]:
        """Register the mailbox against the Pub/Sub topic.

        Expires after 7 days, so the scheduler renews daily. Weekly renewal
        leaves no margin for a single failed run.
        """
        if not self.settings.pubsub_topic:
            raise ValueError("PUBSUB_TOPIC is not configured")
        svc = await self.service()
        body = {"topicName": self.settings.pubsub_topic, "labelFilterBehavior": "include"}
        response = await self._execute(
            svc.users().watch(userId=self.settings.gmail_user_id, body=body), cost=1
        )
        expiry = datetime.fromtimestamp(int(response["expiration"]) / 1000, tz=UTC)
        log.info("watch registered until %s (historyId=%s)", expiry, response.get("historyId"))
        return {"history_id": str(response["historyId"]), "expiry": expiry}

    async def stop_watch(self) -> None:
        svc = await self.service()
        await self._execute(svc.users().stop(userId=self.settings.gmail_user_id), cost=1)
