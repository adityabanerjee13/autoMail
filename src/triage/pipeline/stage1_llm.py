"""Stage 1: the LLM judgment.

Named stage1 because phase 2 puts a supervised classifier in front of it and
demotes this to the low-confidence fallback. Until that data exists, every
message comes through here.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from triage.config import Settings, get_settings
from triage.db import repo
from triage.llm.client import LLMClient, LLMResult
from triage.llm.fewshot import build_examples
from triage.schemas import Message

#: Rough chars-per-token for English prose. Deliberately not a real tokenizer:
#: loading one costs RAM this box does not have to spare, and the budget only
#: needs to be approximately right.
CHARS_PER_TOKEN = 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut the body to the configured budget, on a word boundary."""
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Back up to the last whitespace so the model does not see half a word.
    space = cut.rfind(" ")
    if space > limit * 0.8:
        cut = cut[:space]
    return cut.rstrip() + "\n[truncated]"


async def build_context(
    session: AsyncSession,
    message: Message,
    features: dict[str, Any],
    *,
    embedding: list[float] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Everything the prompt template needs, and nothing else.

    The thread block is snippets only. Sending full quoted history inflates
    latency without improving accuracy -- the dequoted body already contains
    what the sender actually wrote.
    """
    settings = settings or get_settings()
    thread = await repo.thread_context(session, message.thread_id, exclude_id=message.id)
    fewshot = await build_examples(
        session, embedding, exclude_message_id=message.id, settings=settings
    )
    return {
        "from_addr": message.from_addr,
        "to_addrs": ", ".join(message.to_addrs) or "(none)",
        "cc_addrs": ", ".join(message.cc_addrs),
        "subject": message.subject or "(no subject)",
        "body": truncate_to_tokens(message.body_clean or "(empty body)", settings.max_body_tokens),
        "message_date": message.internal_date.date().isoformat(),
        "features": features,
        "thread": thread,
        "fewshot": fewshot,
    }


async def classify(
    client: LLMClient, context: dict[str, Any], *, settings: Settings | None = None
) -> LLMResult:
    settings = settings or get_settings()
    return await client.classify(context, version=settings.prompt_version)
