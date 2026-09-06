"""Thread identity for messages that arrive without a Gmail thread id.

The Gmail API and the X-GM-THRID IMAP extension both hand us a thread id
directly, so this module is a fallback for .eml fixtures and for any source
that lacks Gmail's extensions. It is also where subject normalisation lives,
which the dedup fingerprint reuses.
"""

from __future__ import annotations

import re

#: Reply/forward prefixes across the locales that actually show up in a
#: personal mailbox. Matched repeatedly: "Re: Fwd: Re: ..." is common.
_PREFIX = re.compile(
    r"^\s*(re|fwd?|aw|antw|sv|vs|rif|res|odp|ynt|回复|转发)\s*(\[\d+\])?\s*:\s*",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


def normalize_subject(subject: str | None) -> str:
    """Strip reply/forward prefixes and collapse whitespace."""
    s = (subject or "").strip()
    while True:
        stripped = _PREFIX.sub("", s, count=1)
        if stripped == s:
            break
        s = stripped
    return _WHITESPACE.sub(" ", s).strip()


def parse_references(headers: dict[str, str]) -> list[str]:
    """Message-IDs from References, oldest first, with In-Reply-To appended."""
    refs = headers.get("references", "") or ""
    ids = re.findall(r"<[^<>@\s]+@[^<>\s]+>", refs)
    in_reply_to = re.findall(r"<[^<>@\s]+@[^<>\s]+>", headers.get("in-reply-to", "") or "")
    for mid in in_reply_to:
        if mid not in ids:
            ids.append(mid)
    return ids


def thread_key(
    headers: dict[str, str], subject: str | None, message_id_hdr: str | None
) -> str:
    """Best-effort thread identity when no Gmail thread id is available.

    Root of the References chain if there is one, otherwise this message's own
    Message-ID, otherwise the normalised subject. Never returns empty -- an
    empty thread id would violate the NOT NULL on messages.thread_id.
    """
    refs = parse_references(headers)
    if refs:
        return refs[0]
    if message_id_hdr:
        return message_id_hdr
    normalized = normalize_subject(subject)
    return normalized or "unthreaded"
