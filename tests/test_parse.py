"""Parsing tests: fixtures in, text out. No database, no network.

parse/ is where the subtle bugs live -- a dequoter that eats the new content,
a charset guess that mangles a subject -- and they are invisible downstream
because the pipeline happily classifies the wrong text. Everything here must
stay runnable with no services up.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path

import pytest

from triage.parse.dequote import clean_body, strip_footer, strip_quoted, strip_signature
from triage.parse.html import html_to_text
from triage.parse.mime import gmail_id_from_msgid, parse_eml, parse_gmail_message
from triage.parse.threading import normalize_subject, parse_references, thread_key
from triage.pipeline.dedup import fingerprint, normalize_for_fingerprint
from triage.schemas import Message

FIXTURES = Path(__file__).parent / "fixtures" / "emails"


def load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def parse_fixture(name: str, **kwargs):
    return parse_eml(load(name), gmail_id=name, thread_id="t1", **kwargs)


# ---------------------------------------------------------------------------
# mime
# ---------------------------------------------------------------------------


def test_plain_reply_headers_and_addresses():
    parsed = parse_fixture("reply_with_quote.eml")
    assert parsed.from_addr == "priya@example.com"
    assert parsed.to_addrs == ["owner@example.com"]
    assert parsed.cc_addrs == ["sam@example.com"]
    assert parsed.message_id_hdr == "<CAF3n2mX9@mail.example.com>"
    assert parsed.internal_date == datetime(2026, 3, 2, 18, 22, 4, tzinfo=UTC)


def test_quoted_history_is_removed_but_new_content_survives():
    parsed = parse_fixture("reply_with_quote.eml")
    assert "Saturday works" in parsed.body_clean
    # Everything below the "On ... wrote:" banner is history, not new content.
    assert "Thai place" not in parsed.body_clean
    assert ">" not in parsed.body_clean


def test_rfc2047_encoded_subject_is_decoded():
    parsed = parse_fixture("html_newsletter.eml")
    assert parsed.subject == "The Weekly Dispatch – issue 214"


def test_html_body_is_flattened_and_scripts_dropped():
    parsed = parse_fixture("html_newsletter.eml")
    assert "distributed systems" in parsed.body_clean
    assert "urban planning" in parsed.body_clean
    assert "window.track" not in parsed.body_clean
    assert "color:red" not in parsed.body_clean
    assert "<" not in parsed.body_clean


def test_bulk_headers_are_kept_and_others_dropped():
    parsed = parse_fixture("html_newsletter.eml")
    assert parsed.headers["list-unsubscribe"] == "<https://newsletter.example.io/u/abc123>"
    assert parsed.headers["precedence"] == "bulk"
    assert "mime-version" not in parsed.headers


def test_attachment_is_flagged_not_parsed_as_body():
    parsed = parse_fixture("invoice_with_attachment.eml")
    assert parsed.has_attachments is True
    assert "account ending 4417" in parsed.body_clean
    assert "JVBERi0xLjQ" not in parsed.body_clean


def test_confidentiality_footer_is_stripped():
    parsed = parse_fixture("invoice_with_attachment.eml")
    assert "Minimum payment 25.00" in parsed.body_clean
    assert "confidential" not in parsed.body_clean


def test_parsing_is_deterministic():
    """Same bytes in, same text out -- twice."""
    first = parse_fixture("html_newsletter.eml")
    second = parse_fixture("html_newsletter.eml")
    assert first.body_clean == second.body_clean
    assert first.snippet == second.snippet


def test_missing_date_falls_back_without_raising():
    raw = b"From: a@example.com\nTo: b@example.com\nSubject: no date\n\nbody\n"
    parsed = parse_eml(raw, gmail_id="x", thread_id="t")
    assert parsed.internal_date.tzinfo is not None


def test_malformed_charset_does_not_raise():
    raw = (
        b"From: a@example.com\nTo: b@example.com\nSubject: bad charset\n"
        b'Content-Type: text/plain; charset="definitely-not-a-charset"\n\n'
        b"caf\xe9 latte\n"
    )
    parsed = parse_eml(raw, gmail_id="x", thread_id="t")
    assert "latte" in parsed.body_clean


# ---------------------------------------------------------------------------
# gmail api payloads
# ---------------------------------------------------------------------------


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def test_gmail_payload_parses_to_the_same_shape():
    message = {
        "id": "18f0c2a9b7d",
        "threadId": "18f0c2a9b00",
        "labelIds": ["INBOX", "CATEGORY_UPDATES"],
        "internalDate": "1772616720000",
        "snippet": "Your statement is ready",
        "sizeEstimate": 4211,
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": "Billing <billing@examplebank.com>"},
                {"name": "To", "value": "owner@example.com"},
                {"name": "Subject", "value": "Your statement is ready"},
                {"name": "Message-ID", "value": "<abc@examplebank.com>"},
                {"name": "X-Whatever", "value": "dropped"},
            ],
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64("Statement ready. Pay by 14 March.")},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": _b64("<p>Statement ready.</p>")},
                },
            ],
        },
    }
    parsed = parse_gmail_message(message)
    assert parsed.gmail_id == "18f0c2a9b7d"
    assert parsed.from_addr == "billing@examplebank.com"
    assert parsed.labels == ["INBOX", "CATEGORY_UPDATES"]
    # text/plain wins over text/html when both are present.
    assert "Pay by 14 March" in parsed.body_clean
    assert "x-whatever" not in parsed.headers


def test_gmail_id_is_hex_of_x_gm_msgid():
    """This equivalence is what lets the IMAP backfill and the API sync
    deduplicate against each other."""
    assert gmail_id_from_msgid(1729382256910270464) == "1800000000000000"
    assert gmail_id_from_msgid("255") == "ff"


# ---------------------------------------------------------------------------
# dequoting units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "New text.\n\nOn Mon, 2 Mar 2026 at 17:04, Owner <o@e.com> wrote:\n> old",
        "New text.\n\n-----Original Message-----\nFrom: someone\nold",
        "New text.\n\n________________________________\nFrom: someone\nold",
        "New text.\n\n> quoted line one\n> quoted line two",
    ],
)
def test_quote_banners(body):
    assert strip_quoted(body).strip() == "New text."


def test_signature_only_stripped_from_the_tail():
    body = "Thanks for the update.\n\nI will look tomorrow.\n\nBest regards,\nPriya"
    out = strip_signature(body)
    # The opening "Thanks" is a greeting, not a sign-off.
    assert "Thanks for the update." in out
    assert "Priya" not in out


def test_footer_at_line_zero_keeps_the_body():
    """A body that is nothing but boilerplate still has to reach the model."""
    body = "Unsubscribe here if you no longer wish to receive these."
    assert strip_footer(body) == body


def test_clean_body_is_idempotent():
    body = "Hi.\n\nOn Mon someone wrote:\n> old\n\n--\nSig"
    once = clean_body(body)
    assert clean_body(once) == once


# ---------------------------------------------------------------------------
# html
# ---------------------------------------------------------------------------


def test_block_elements_become_line_breaks():
    text = html_to_text("<p>one</p><p>two</p><div>three</div>")
    assert text.split("\n")[0] == "one"
    assert "two" in text and "three" in text


def test_inline_elements_do_not_break_words():
    assert html_to_text("<p>pay <b>now</b> please</p>") == "pay now please"


def test_entities_are_unescaped():
    assert "&" in html_to_text("<p>Tom &amp; Jerry</p>")


# ---------------------------------------------------------------------------
# threading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subject,expected",
    [
        ("Re: dinner", "dinner"),
        ("RE: FWD: Re: dinner", "dinner"),
        ("Fw: dinner", "dinner"),
        ("AW: dinner", "dinner"),
        ("Re[2]: dinner", "dinner"),
        ("dinner", "dinner"),
    ],
)
def test_normalize_subject(subject, expected):
    assert normalize_subject(subject) == expected


def test_thread_key_prefers_the_root_of_the_references_chain():
    headers = {"references": "<root@x> <second@x>", "in-reply-to": "<second@x>"}
    assert thread_key(headers, "Re: hi", "<third@x>") == "<root@x>"
    assert parse_references(headers) == ["<root@x>", "<second@x>"]


def test_thread_key_never_returns_empty():
    assert thread_key({}, None, None) == "unthreaded"


# ---------------------------------------------------------------------------
# fingerprinting (pure text normalisation, so it lives with the parse tests)
# ---------------------------------------------------------------------------


def _message(subject: str, sender: str = "no-reply@shop.example") -> Message:
    return Message(
        id=1,
        gmail_id="g1",
        thread_id="t1",
        from_addr=sender,
        subject=subject,
        internal_date=datetime(2026, 3, 1, tzinfo=UTC),
        created_at=datetime(2026, 3, 1, tzinfo=UTC),
    )


def test_same_template_different_order_number_matches():
    a = fingerprint(_message("Your order #88213 has shipped"))
    b = fingerprint(_message("Your order #90551 has shipped"))
    assert a is not None and a == b


def test_same_subject_different_sender_does_not_match():
    a = fingerprint(_message("Your monthly statement is ready"))
    b = fingerprint(_message("Your monthly statement is ready", sender="other@bank.example"))
    assert a != b


def test_different_templates_do_not_collide():
    a = fingerprint(_message("Your order has shipped"))
    b = fingerprint(_message("Your payment failed"))
    assert a != b


def test_generic_subject_gets_no_fingerprint():
    """No key is better than a key that collides across every sender."""
    assert fingerprint(_message("Hi")) is None
    assert fingerprint(_message("#12345")) is None


def test_dates_and_amounts_are_normalised_away():
    assert normalize_for_fingerprint("Statement for 2026-03-01: £842.19 due") == (
        normalize_for_fingerprint("Statement for 2026-04-01: £901.00 due")
    )
