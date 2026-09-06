"""Remove quoted reply history and signature blocks.

talon does this better than a pile of regexes, but it drags in lxml and has a
history of breaking on new CPython releases, so it is an optional extra. When
it is missing we fall back to the heuristics below, which handle the common
client formats (Gmail, Outlook, Apple Mail) and are what the fixtures test.

Both paths are pure functions over a string.
"""

from __future__ import annotations

import re

try:  # pragma: no cover - import guard
    from talon import quotations as _talon_quotations
    from talon.signature.bruteforce import extract_signature as _talon_signature

    _HAVE_TALON = True
except Exception:  # pragma: no cover - talon import can fail at runtime too
    _talon_quotations = None
    _talon_signature = None
    _HAVE_TALON = False


#: Client-specific banners that introduce quoted history. Everything from the
#: first match onward is reply history, not new content.
_QUOTE_HEADERS = [
    re.compile(r"^\s*On .{0,120}\bwrote:\s*$", re.IGNORECASE),
    re.compile(r"^\s*On .{0,80},.{0,80}\bwrote:", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Forwarded message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*_{5,}\s*$"),
    re.compile(r"^\s*From:\s*.+$", re.IGNORECASE),
    re.compile(r"^\s*Sent from my \w+", re.IGNORECASE),
    re.compile(r"^\s*在.{0,80}写道：\s*$"),  # zh "wrote:"
]

#: A run of these many consecutive '>' lines is treated as quoted history.
_QUOTE_PREFIX = re.compile(r"^\s*>")

_SIGNATURE_MARKERS = [
    re.compile(r"^\s*--\s*$"),
    re.compile(r"^\s*—\s*$"),
    re.compile(r"^\s*Best regards,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Kind regards,?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(Thanks|Cheers|Regards|Sincerely|Best),?\s*$", re.IGNORECASE),
]

#: Bulk-mail footers. Cutting these keeps the token budget for actual content.
_FOOTER_MARKERS = [
    re.compile(r"^\s*(To )?unsubscribe\b", re.IGNORECASE),
    re.compile(r"^\s*You (are )?receiv(ed|ing) this (e-?mail|message)", re.IGNORECASE),
    re.compile(r"^\s*This (e-?mail|message) (and any attachments )?is confidential", re.IGNORECASE),
    re.compile(r"^\s*View this email in your browser", re.IGNORECASE),
    re.compile(r"^\s*Manage (your )?(email )?preferences", re.IGNORECASE),
]


def strip_quoted(text: str) -> str:
    """Drop quoted reply history, keeping only the newly written portion."""
    if not text:
        return ""
    if _HAVE_TALON:
        try:
            return _talon_quotations.extract_from_plain(text).strip()
        except Exception:  # pragma: no cover - talon raises on odd input
            pass
    return _strip_quoted_heuristic(text)


def _strip_quoted_heuristic(text: str) -> str:
    lines = text.split("\n")
    cut = len(lines)

    for i, line in enumerate(lines):
        # A "From:" line only means quoted history when it is not the first
        # thing in the body -- some newsletters legitimately open with one.
        if i == 0:
            continue
        if any(p.match(line) for p in _QUOTE_HEADERS):
            cut = i
            break
        # Two consecutive '>' lines: quoted block with no banner (Outlook).
        if _QUOTE_PREFIX.match(line) and i + 1 < len(lines) and _QUOTE_PREFIX.match(lines[i + 1]):
            cut = i
            break

    kept = lines[:cut]
    # Drop a trailing "On ... wrote:" fragment split across two lines.
    while kept and not kept[-1].strip():
        kept.pop()
    if kept and kept[-1].rstrip().endswith(("wrote:", "wrote :")):
        kept.pop()
    return "\n".join(kept).strip()


def strip_signature(text: str) -> str:
    """Drop a trailing signature block."""
    if not text:
        return ""
    if _HAVE_TALON:
        try:
            body, _sig = _talon_signature(text, sender="")
            return body.strip()
        except Exception:  # pragma: no cover
            pass
    return _strip_signature_heuristic(text)


def _strip_signature_heuristic(text: str) -> str:
    lines = text.split("\n")
    # Only look in the tail: a "Thanks," in the first paragraph of a long mail
    # is a greeting, not a sign-off.
    window = max(len(lines) - 12, 0)
    for i in range(len(lines) - 1, window - 1, -1):
        if any(p.match(lines[i]) for p in _SIGNATURE_MARKERS):
            return "\n".join(lines[:i]).strip()
    return text.strip()


def strip_footer(text: str) -> str:
    """Drop bulk-mail boilerplate from the tail of the body."""
    if not text:
        return ""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if any(p.match(line) for p in _FOOTER_MARKERS):
            # Keep at least something: a footer at line 0 means the whole body
            # is boilerplate and the subject is all we have.
            return "\n".join(lines[:i]).strip() if i else text.strip()
    return text.strip()


def clean_body(text: str) -> str:
    """The full body-cleaning chain used by the ingest path."""
    out = strip_quoted(text)
    out = strip_signature(out)
    out = strip_footer(out)
    return out.strip()
