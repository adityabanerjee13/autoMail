"""Template fingerprinting.

A personal mailbox is mostly the same twenty templates over and over: the same
receipt, the same alert, the same digest with a different date and order
number. Fingerprinting normalises those varying parts away so the second
instance of a template can copy the first judgment instead of spending GPU
time on it.

Deliberately conservative. A false fingerprint match writes a wrong label with
no LLM call behind it, which is the most expensive error this module can make,
so only a high-confidence prior judgment is ever copied.
"""

from __future__ import annotations

import hashlib
import re

from triage.parse.threading import normalize_subject
from triage.schemas import Classification, Message

#: Everything that varies between instances of one template.
_VARIABLE = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), " "),                     # ISO dates
    (re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"), " "),       # numeric dates
    (re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?\b", re.I), " "),  # times
    (re.compile(r"[#№]\s*\w*\d+\w*"), " "),                          # order numbers
    (re.compile(r"\b[\w.+-]+@[\w.-]+\b"), " "),                      # addresses
    (re.compile(r"https?://\S+"), " "),                              # urls
    (re.compile(r"[£$€₹¥]\s?[\d,]+(\.\d+)?"), " "),                  # amounts
    (re.compile(r"\d+"), " "),                                       # any remaining digits
    (re.compile(r"[^\w\s]"), " "),                                   # punctuation
    (re.compile(r"\s+"), " "),
]

#: Below this many characters the normalised subject is not distinctive enough
#: to key on -- "your order" would collide across every retailer.
_MIN_SUBJECT_CHARS = 8


def normalize_for_fingerprint(subject: str | None) -> str:
    text = normalize_subject(subject).lower()
    for pattern, repl in _VARIABLE:
        text = pattern.sub(repl, text)
    return text.strip()


def fingerprint(message: Message) -> str | None:
    """Stable key for "this is the same template from the same sender".

    Returns None when the subject normalises to something too generic to be a
    safe key -- no fingerprint is better than a colliding one.
    """
    normalized = normalize_for_fingerprint(message.subject)
    if len(normalized) < _MIN_SUBJECT_CHARS:
        return None
    sender = message.from_addr.strip().lower()
    digest = hashlib.sha256(f"{sender}\x00{normalized}".encode()).hexdigest()
    return digest[:32]


def is_copyable(prior: Classification) -> bool:
    """Whether a prior judgment is trustworthy enough to copy without an LLM call.

    A human correction is always copyable. An LLM judgment must have been
    confident on both axes -- copying a "low" would propagate a guess and,
    worse, keep it out of the review queue.
    """
    if prior.source == "human":
        return True
    if prior.source != "llm":
        # Never chain dedup off another dedup: one bad original would spread
        # across the whole mailbox.
        return False
    payload = prior.payload or {}
    return (
        payload.get("importance_confidence") == "high"
        and payload.get("category_confidence") == "high"
    )
