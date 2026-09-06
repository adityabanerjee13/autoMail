"""Few-shot example retrieval over already-labelled messages.

Config-gated and off by default. On day one the pool is empty, so this returns
nothing and only costs a pgvector query; leave ENABLE_FEWSHOT false until
several hundred human corrections exist. Examples the user corrected are worth
far more than examples the model produced, so they sort first.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from triage.config import Settings, get_settings
from triage.db import repo

log = logging.getLogger(__name__)


async def build_examples(
    session: AsyncSession,
    embedding: list[float] | None,
    *,
    exclude_message_id: int,
    settings: Settings | None = None,
) -> list[dict[str, str]]:
    """Nearest labelled neighbours, shaped for the prompt template."""
    settings = settings or get_settings()
    if not settings.enable_fewshot or not embedding:
        return []

    pairs = await repo.nearest_labelled(
        session,
        embedding,
        k=settings.fewshot_k,
        exclude_message_id=exclude_message_id,
        prefer_human=True,
    )
    examples: list[dict[str, str]] = []
    for message, classification in pairs:
        # Only the decision fields go in the example. Showing the model a
        # `reason` written by an earlier version of itself teaches it to
        # imitate its own justifications rather than judge the message.
        label = {
            "is_important": classification.is_important,
            "category": classification.category,
            "action_required": bool(classification.payload.get("action_required")),
        }
        examples.append(
            {
                "from_addr": message.from_addr,
                "subject": message.subject or "(no subject)",
                "snippet": (message.snippet or message.body_clean or "")[:300],
                "label_json": json.dumps(label),
            }
        )
    log.debug("fewshot: %d examples for message %s", len(examples), exclude_message_id)
    return examples
