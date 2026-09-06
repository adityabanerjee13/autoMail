"""RFC 822 and Gmail API payloads to a single normalised shape.

Both ingest sources converge here so that nothing downstream can tell whether a
message arrived over the API or over IMAP. The Gmail message id is the hex form
of ``X-GM-MSGID``, which is what lets the IMAP backfill and the API steady state
share one ``gmail_id`` key and deduplicate against each other.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email import message_from_bytes, policy
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import getaddresses, parsedate_to_datetime

from triage.parse.dequote import clean_body
from triage.parse.html import html_to_text, tidy

#: Only these headers are persisted. The full set is large, mostly noise, and
#: occasionally contains things worth not storing.
KEPT_HEADERS = (
    "message-id",
    "in-reply-to",
    "references",
    "reply-to",
    "return-path",
    "delivered-to",
    "date",
    "list-id",
    "list-unsubscribe",
    "precedence",
    "auto-submitted",
    "x-mailer",
    "x-priority",
    "importance",
    "authentication-results",
    "content-type",
)

SNIPPET_LEN = 300


@dataclass(slots=True)
class ParsedEmail:
    gmail_id: str
    thread_id: str
    message_id_hdr: str | None
    from_addr: str
    to_addrs: list[str]
    cc_addrs: list[str]
    subject: str | None
    body_clean: str
    snippet: str
    labels: list[str]
    headers: dict[str, str]
    internal_date: datetime
    has_attachments: bool = False
    raw_size: int = 0
    extra: dict = field(default_factory=dict)


def gmail_id_from_msgid(x_gm_msgid: int | str) -> str:
    """X-GM-MSGID (decimal) -> the Gmail API message id (lowercase hex)."""
    return format(int(x_gm_msgid), "x")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        # Malformed RFC 2047 encoding is common in spam; keep the raw bytes
        # rather than dropping the message.
        return value.strip()


def _addresses(value: str | None) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for _name, addr in getaddresses([value]):
        addr = addr.strip().lower()
        if addr and "@" in addr:
            out.append(addr)
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(out))


def _first_address(value: str | None) -> str:
    addrs = _addresses(value)
    return addrs[0] if addrs else (value or "").strip().lower()


def _snippet(body: str) -> str:
    flat = " ".join(body.split())
    return flat[:SNIPPET_LEN]


def _b64url(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data + pad)
    except (binascii.Error, ValueError):
        return b""


# ---------------------------------------------------------------------------
# .eml / IMAP
# ---------------------------------------------------------------------------


def parse_eml(
    raw: bytes,
    *,
    gmail_id: str,
    thread_id: str,
    labels: list[str] | None = None,
    internal_date: datetime | None = None,
) -> ParsedEmail:
    msg: EmailMessage = message_from_bytes(raw, policy=policy.default)  # type: ignore[assignment]

    plain, html, has_attachments = _extract_parts(msg)
    body = plain if plain.strip() else html_to_text(html)
    body = clean_body(tidy(body))

    hdrs = {k.lower(): _decode_header_value(v) for k, v in msg.items() if k.lower() in KEPT_HEADERS}

    date = internal_date or _header_date(msg)
    return ParsedEmail(
        gmail_id=gmail_id,
        thread_id=thread_id,
        message_id_hdr=hdrs.get("message-id") or None,
        from_addr=_first_address(msg.get("From")),
        to_addrs=_addresses(msg.get("To")),
        cc_addrs=_addresses(msg.get("Cc")),
        subject=_decode_header_value(msg.get("Subject")) or None,
        body_clean=body,
        snippet=_snippet(body),
        labels=labels or [],
        headers=hdrs,
        internal_date=date,
        has_attachments=has_attachments,
        raw_size=len(raw),
    )


def _header_date(msg: EmailMessage) -> datetime:
    raw = msg.get("Date")
    if raw:
        try:
            dt = parsedate_to_datetime(raw)
            if dt is not None:
                return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            pass
    # A message with no parseable Date still has to land somewhere on the
    # timeline; ingestion time is the least wrong answer.
    return datetime.now(UTC)


def _extract_parts(msg: EmailMessage) -> tuple[str, str, bool]:
    plain: list[str] = []
    html: list[str] = []
    has_attachments = False

    for part in msg.walk():
        if part.is_multipart():
            continue
        disposition = (part.get_content_disposition() or "").lower()
        if disposition == "attachment":
            has_attachments = True
            continue
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            has_attachments = has_attachments or bool(disposition)
            continue
        try:
            content = part.get_content()
        except (LookupError, UnicodeDecodeError):
            # Unknown or lying charset. latin-1 never raises and keeps the
            # ASCII subset intact, which is enough to classify on.
            payload = part.get_payload(decode=True) or b""
            content = payload.decode("latin-1", errors="replace")
        if ctype == "text/plain":
            plain.append(content)
        else:
            html.append(content)

    return "\n".join(plain), "\n".join(html), has_attachments


# ---------------------------------------------------------------------------
# Gmail API
# ---------------------------------------------------------------------------


def parse_gmail_message(message: dict) -> ParsedEmail:
    """Parse a users.messages.get response in format='full'."""
    payload = message.get("payload", {}) or {}
    hdr_list = payload.get("headers", []) or []
    raw_headers = {h.get("name", "").lower(): h.get("value", "") for h in hdr_list}
    hdrs = {k: _decode_header_value(v) for k, v in raw_headers.items() if k in KEPT_HEADERS}

    plain, html, has_attachments = _walk_gmail_parts(payload)
    body = plain if plain.strip() else html_to_text(html)
    body = clean_body(tidy(body))

    internal = message.get("internalDate")
    date = (
        datetime.fromtimestamp(int(internal) / 1000, tz=UTC)
        if internal
        else datetime.now(UTC)
    )

    return ParsedEmail(
        gmail_id=message["id"],
        thread_id=message.get("threadId", message["id"]),
        message_id_hdr=hdrs.get("message-id") or None,
        from_addr=_first_address(raw_headers.get("from")),
        to_addrs=_addresses(raw_headers.get("to")),
        cc_addrs=_addresses(raw_headers.get("cc")),
        subject=_decode_header_value(raw_headers.get("subject")) or None,
        body_clean=body,
        # Gmail's own snippet is HTML-unescaped and already trimmed; prefer it
        # when present so the UI matches what Gmail shows.
        snippet=message.get("snippet") or _snippet(body),
        labels=list(message.get("labelIds", []) or []),
        headers=hdrs,
        internal_date=date,
        has_attachments=has_attachments,
        raw_size=int(message.get("sizeEstimate", 0) or 0),
    )


def _walk_gmail_parts(payload: dict) -> tuple[str, str, bool]:
    plain: list[str] = []
    html: list[str] = []
    has_attachments = False

    stack = [payload]
    while stack:
        part = stack.pop()
        mime = part.get("mimeType", "")
        if part.get("parts"):
            stack.extend(part["parts"])
            continue
        body = part.get("body", {}) or {}
        if part.get("filename"):
            has_attachments = True
            continue
        data = body.get("data")
        if not data:
            # attachmentId without data means the body must be fetched
            # separately; not worth a quota unit for classification.
            continue
        text = _b64url(data).decode("utf-8", errors="replace")
        if mime == "text/plain":
            plain.append(text)
        elif mime == "text/html":
            html.append(text)

    return "\n".join(plain), "\n".join(html), has_attachments
