"""Rendering tool results for the model. Compact text, never JSON.

Two reasons this is not ``json.dumps``. The obvious one is size: a
pretty-printed array of ten message objects costs roughly twice what the same
ten rows cost as pipe-delimited lines, and with a 6,330-token prompt budget
that difference is two extra tool calls' worth of room. The less obvious one is
that a model asked to emit JSON does better when the *only* JSON in its context
is its own output; feeding it JSON results encourages it to echo their shape.

Everything here truncates. A mailbox has 4,000-character bodies and 200-
character subjects in it, and one ungoverned field can eat a whole step's
budget.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

#: Widths tuned so a full row lands near 100 characters -- roughly 33 tokens.
#: Ten rows is then ~350 tokens, which fits the per-result cap with room for
#: the header.
SUBJECT_W = 70
SENDER_W = 38
BODY_W = 1200


def clip(value: Any, width: int) -> str:
    """One line, at most ``width`` characters, with an explicit ellipsis.

    Newlines are stripped rather than escaped. A subject containing a newline is
    rare but real, and one of them would otherwise split a row in half and
    silently corrupt every row after it in the model's reading of the table.
    """
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def short_date(value: datetime | None, *, now: datetime | None = None) -> str:
    """``MM-DD``, or ``YYYY-MM-DD`` when the year is not the current one.

    The year is dead weight on a mailbox that is mostly recent, and misleading
    by its absence on the mail that is not.
    """
    if value is None:
        return "?"
    local = value.astimezone()
    current = (now or datetime.now(local.tzinfo)).year
    return local.strftime("%m-%d" if local.year == current else "%Y-%m-%d")


def render_rows(rows: list, *, total: int | None = None, tool: str = "") -> str:
    """A message table: header line, column names, one line per message.

    ``rows`` are ``MessageWithClassification``. The count line matters more than
    it looks: "(5 of 23)" is what stops the model reporting five results as if
    they were all of them.
    """
    if not rows:
        return "no messages matched."
    head = f"{len(rows)} shown" + (f" of {total}" if total is not None else "")
    lines = [f"{head}", "id | date | from | subject | category | important"]
    for row in rows:
        m = row.message
        c = row.classification
        lines.append(
            " | ".join(
                (
                    str(m.id),
                    short_date(m.internal_date),
                    clip(m.from_addr, SENDER_W),
                    clip(m.subject or "(no subject)", SUBJECT_W),
                    c.category if c else "-",
                    ("yes" if c.is_important else "no") if c else "-",
                )
            )
        )
    return "\n".join(lines)


def render_unprocessed(rows: list[dict]) -> str:
    """``repo.unprocessed`` returns plain dicts, not models -- its own renderer."""
    if not rows:
        return "nothing unprocessed: every message is classified at the current model and prompt."
    lines = ["id | date | from | subject | job"]
    for r in rows:
        lines.append(
            " | ".join(
                (
                    str(r["id"]),
                    short_date(r["internal_date"]),
                    clip(r["from_addr"], SENDER_W),
                    clip(r["subject"] or "(no subject)", SUBJECT_W),
                    r["job_status"] or "not queued",
                )
            )
        )
    return "\n".join(lines)


def render_message(message, classification) -> str:
    """One message in full, for ``read_message``.

    The body is the only unbounded thing the agent can pull into context, so it
    is capped hard. 1,200 characters is roughly 400 tokens -- enough to answer
    "what does this say" for all but the longest mail, and small enough that
    reading three messages in one turn still fits.
    """
    parts = [
        f"id: {message.id}",
        f"date: {message.internal_date.astimezone().strftime('%Y-%m-%d %H:%M')}",
        f"from: {clip(message.from_addr, 120)}",
        f"to: {clip(', '.join(message.to_addrs or []), 120)}",
        f"subject: {clip(message.subject or '(no subject)', 200)}",
    ]
    if classification is None:
        parts.append("classification: none yet")
    else:
        p = classification.payload or {}
        parts.append(
            f"classification: {classification.category}, "
            f"{'important' if classification.is_important else 'not important'}, "
            f"source {classification.source}"
        )
        if p.get("reason"):
            parts.append(f"reason: {clip(p['reason'], 200)}")
        if p.get("deadline"):
            parts.append(f"deadline: {p['deadline']}")
    body = (message.body_clean or "").strip()
    if len(body) > BODY_W:
        body = body[:BODY_W] + "\n[body truncated]"
    parts.append("body:\n" + (body or "(empty)"))
    return "\n".join(parts)
