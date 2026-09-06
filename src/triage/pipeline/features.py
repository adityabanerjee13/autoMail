"""Cheap signals passed into the prompt as context.

These are not a rules stage. Nothing here decides anything -- every message
still goes to the LLM. They exist because "the recipient has replied to this
sender before" and "this arrived on a mailing list" are facts the model cannot
see from the message text alone.
"""

from __future__ import annotations

import re

from sqlalchemy.ext.asyncio import AsyncSession

from triage.db import repo
from triage.schemas import Message

#: Local parts that indicate nobody is reading replies.
_AUTOMATED_LOCAL = re.compile(
    r"^(no-?reply|do-?not-?reply|notifications?|mailer|bounce|postmaster|automated|"
    r"alerts?|updates?|news|info|support|billing|receipts?)([+.-]|$)",
    re.IGNORECASE,
)

#: Headers that only appear on bulk mail.
_BULK_HEADERS = ("list-id", "list-unsubscribe", "precedence", "auto-submitted")


def recipient_position(message: Message) -> str:
    """To, Cc, or neither -- 'neither' means Bcc or a mailing list.

    Determined against Delivered-To, which is the address the message was
    actually delivered to, rather than any configured identity. Aliases and
    plus-addressing therefore work without extra configuration.
    """
    own = (message.headers.get("delivered-to") or "").strip().lower()
    if not own:
        return "unknown"
    if own in {a.lower() for a in message.to_addrs}:
        return "To"
    if own in {a.lower() for a in message.cc_addrs}:
        return "Cc"
    return "Bcc or mailing list"


def bulk_headers(message: Message) -> list[str]:
    return [h for h in _BULK_HEADERS if message.headers.get(h)]


def is_automated_sender(from_addr: str) -> bool:
    local = from_addr.split("@", 1)[0] if "@" in from_addr else from_addr
    return bool(_AUTOMATED_LOCAL.match(local))


async def build(session: AsyncSession, message: Message) -> dict:
    """Assemble the feature block the prompt template renders."""
    history = await repo.sender_features(
        session, message.from_addr, before=message.internal_date
    )
    return {
        **history,
        "recipient_position": recipient_position(message),
        "bulk_headers": ", ".join(bulk_headers(message)) or "none",
        "automated_sender": is_automated_sender(message.from_addr),
        # Gmail's own labels are a signal in their own right: CATEGORY_PROMOTIONS
        # is Google's opinion, not ours, but it is worth showing the model.
        "gmail_labels": ", ".join(message.labels) or "none",
        "has_attachments": message.headers.get("content-type", "").startswith(
            "multipart/mixed"
        ),
    }
