"""Pure, deterministic parsing: same bytes in, same text out.

No network, no database, no configuration. Everything here must be unit
testable against .eml fixtures alone -- this is where the subtle bugs live.
"""

from triage.parse.dequote import strip_quoted, strip_signature
from triage.parse.html import html_to_text
from triage.parse.mime import ParsedEmail, parse_eml, parse_gmail_message
from triage.parse.threading import normalize_subject, thread_key

__all__ = [
    "ParsedEmail",
    "html_to_text",
    "normalize_subject",
    "parse_eml",
    "parse_gmail_message",
    "strip_quoted",
    "strip_signature",
    "thread_key",
]
