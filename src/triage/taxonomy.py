"""Single source of truth for the category taxonomy.

Three consumers derive from this module and nothing else:

  1. ``llm/schemas.py``  -- the Pydantic ``Triage`` model
  2. the JSON Schema vLLM constrains generation with (via that same model)
  3. the review UI dropdown (``api/routes/review.py``)

If any consumer hardcodes its own copy of the list, drift produces silent
misclassification that is very hard to debug. Add a category here and it
appears everywhere -- but bump ``SCHEMA_VERSION`` when you do, because rows
labelled under the old taxonomy are not comparable to rows labelled under the
new one.

The descriptions are not documentation. They are rendered verbatim into the
classification prompt, so edit them the way you would edit a prompt: sharp
boundaries, explicit disambiguation against the neighbouring category.
"""

from __future__ import annotations

from enum import StrEnum


class Category(StrEnum):
    PERSONAL = "personal"
    WORK = "work"
    FINANCE = "finance"
    PURCHASE = "purchase"
    TRAVEL = "travel"
    CALENDAR = "calendar"
    ACCOUNT = "account"
    NEWSLETTER = "newsletter"
    PROMOTION = "promotion"
    SOCIAL = "social"
    NOTIFICATION = "notification"
    SPAM = "spam"
    OTHER = "other"


#: Rendered into the prompt, one line per category. Keep each definition to a
#: single sentence plus an explicit boundary against the category it is most
#: often confused with.
CATEGORY_DESCRIPTIONS: dict[Category, str] = {
    Category.PERSONAL: (
        "Written by a human being to the recipient personally -- friends, family, "
        "individual correspondence. Not work matters; those are work even when the "
        "sender is a person."
    ),
    Category.WORK: (
        "Employment, clients, colleagues, contracts, job applications, recruiters. "
        "A human sender discussing professional matters. Meeting invitations go to "
        "calendar instead."
    ),
    Category.FINANCE: (
        "Money owed, held, or moved: bills, invoices, bank and card statements, "
        "payment confirmations, tax, insurance, payroll. A receipt for a specific "
        "thing that was bought is purchase."
    ),
    Category.PURCHASE: (
        "A specific order the recipient placed: order confirmation, shipping and "
        "delivery notices, returns, e-commerce receipts. Recurring bills and "
        "statements are finance."
    ),
    Category.TRAVEL: (
        "Flights, trains, hotels, car hire, itineraries, check-in and gate notices. "
        "A booking receipt for travel is still travel, not purchase."
    ),
    Category.CALENDAR: (
        "Meeting invitations, reschedules, cancellations, reminders for a scheduled "
        "event, and .ics attachments. Use this even when the sender is a colleague."
    ),
    Category.ACCOUNT: (
        "Account and security lifecycle: sign-in alerts, verification codes, "
        "password resets, MFA, terms-of-service and privacy policy changes. Not "
        "product announcements; those are notification or promotion."
    ),
    Category.NEWSLETTER: (
        "Subscribed editorial content sent on a schedule -- digests, publications, "
        "mailing lists. Content the recipient signed up to read, not something "
        "being sold to them."
    ),
    Category.PROMOTION: (
        "Marketing intended to drive a purchase: sales, discounts, offers, "
        "abandoned-cart nudges, upgrade pitches. If the primary purpose is selling, "
        "it is promotion even when dressed as a newsletter."
    ),
    Category.SOCIAL: (
        "Activity on a social network, forum, or community platform: mentions, "
        "follows, replies, connection requests, group digests."
    ),
    Category.NOTIFICATION: (
        "Automated operational messages from a service the recipient uses: CI "
        "results, monitoring alerts, app notifications, system reports. Use this "
        "when the message is machine-generated and fits no more specific category."
    ),
    Category.SPAM: (
        "Unsolicited bulk mail, phishing, scams, and fraudulent impersonation. "
        "Legitimate marketing the recipient could unsubscribe from is promotion."
    ),
    Category.OTHER: (
        "Genuinely does not fit any category above. Prefer a specific category; a "
        "rising rate of other means the taxonomy needs a new entry."
    ),
}

# Fail loudly at import time rather than shipping a prompt with a category the
# model is never told about.
_missing = set(Category) - set(CATEGORY_DESCRIPTIONS)
if _missing:  # pragma: no cover - guards against an incomplete edit
    raise RuntimeError(f"CATEGORY_DESCRIPTIONS is missing: {sorted(_missing)}")


def category_values() -> list[str]:
    """Ordered category strings, for the review UI dropdown."""
    return [c.value for c in Category]


def render_taxonomy() -> str:
    """The taxonomy block injected into the classification prompt."""
    return "\n".join(f"- {c.value}: {CATEGORY_DESCRIPTIONS[c]}" for c in Category)
