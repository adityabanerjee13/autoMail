"""The pipeline, executed per message.

Keep this function boring. If it starts accumulating branching logic the stage
boundaries are wrong -- fix the boundaries. Do not reach for an orchestration
framework: the pipeline is deterministic, the stages are plain functions, and
Pydantic here is validation, not control flow.

Stages, in order:

  1. load               6. few-shot retrieval (config-gated)
  2. idempotency guard  7. LLM call + validate
  3. dedup check        8. persist
  4. features           9. post-actions
  5. prepare input     10. complete / retry / dead-letter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from triage.config import Settings, get_settings
from triage.db import queue, repo
from triage.llm.client import LLMClient, LLMSchemaError
from triage.llm.embeddings import embed, embedding_text
from triage.pipeline import dedup, features, stage1_llm
from triage.schemas import Classification, ClassificationIn, Job
from triage.taxonomy import Category

log = logging.getLogger(__name__)

Status = Literal["classified", "deduped", "skipped", "missing", "failed", "dead"]


@dataclass(slots=True)
class RunOutcome:
    status: Status
    message_id: int
    classification: Classification | None = None
    needs_review: bool = False
    error: str | None = None


async def run_message(
    session: AsyncSession,
    message_id: int,
    *,
    client: LLMClient,
    settings: Settings | None = None,
) -> RunOutcome:
    """Stages 1-9, in one transaction owned by the caller."""
    settings = settings or get_settings()

    # 1. load ---------------------------------------------------------------
    message = await repo.get_message(session, message_id)
    if message is None:
        # The job outlived its message. Nothing to retry.
        return RunOutcome(status="missing", message_id=message_id)

    # 2. idempotency guard --------------------------------------------------
    if await repo.has_classification(
        session,
        message_id,
        model_id=settings.vllm_model_id,
        prompt_version=settings.prompt_version,
    ):
        log.debug("message %s already classified at %s", message_id, settings.prompt_version)
        return RunOutcome(status="skipped", message_id=message_id)

    # 3. dedup check --------------------------------------------------------
    fp = dedup.fingerprint(message)
    if settings.enable_dedup and fp:
        prior = await repo.dedup_candidate(
            session,
            fp,
            model_id=settings.vllm_model_id,
            prompt_version=settings.prompt_version,
            exclude_message_id=message_id,
        )
        if prior is not None and dedup.is_copyable(prior):
            copied = await _persist_copy(session, message_id, prior, settings)
            await _post_actions(session, message, fingerprint=fp, settings=settings)
            return RunOutcome(
                status="deduped",
                message_id=message_id,
                classification=copied,
                needs_review=False,
            )

    # 4. features -----------------------------------------------------------
    feature_block = await features.build(session, message)

    # 6. few-shot needs the embedding, so compute it early when it is enabled;
    #    otherwise it is computed once in post-actions.
    embedding: list[float] | None = None
    if settings.enable_fewshot:
        embedding = await _embed_message(message, settings)

    # 5. prepare input (+ 6. few-shot retrieval) ----------------------------
    context = await stage1_llm.build_context(
        session, message, feature_block, embedding=embedding, settings=settings
    )

    # 7. LLM call + validate ------------------------------------------------
    try:
        result = await stage1_llm.classify(client, context, settings=settings)
    except LLMSchemaError as exc:
        # Persist what came back before dead-lettering: a truncated generation
        # is only debuggable from the raw text.
        log.warning("schema failure on message %s: %s", message_id, exc)
        raise

    # 8. persist ------------------------------------------------------------
    stored = await repo.insert_classification(
        session,
        ClassificationIn(
            message_id=message_id,
            is_important=result.triage.is_important,
            category=result.triage.category.value,
            payload=result.triage.model_dump(mode="json"),
            raw_response=result.raw_response,
            model_id=settings.vllm_model_id,
            prompt_version=settings.prompt_version,
            schema_version=settings.schema_version,
            source="llm",
            input_tokens=result.input_tokens,
            latency_ms=result.latency_ms,
        ),
    )

    # 9. post-actions -------------------------------------------------------
    await _post_actions(
        session, message, fingerprint=fp, embedding=embedding, settings=settings
    )

    needs_review = result.triage.needs_review
    if needs_review:
        log.info(
            "message %s flagged for review (%s/%s): %s",
            message_id,
            result.triage.importance_confidence,
            result.triage.category_confidence,
            result.triage.reason,
        )
    return RunOutcome(
        status="classified",
        message_id=message_id,
        classification=stored,
        needs_review=needs_review,
    )


async def run_job(
    sessions: async_sessionmaker[AsyncSession],
    job: Job,
    *,
    client: LLMClient,
    settings: Settings | None = None,
) -> RunOutcome:
    """Stage 10: run a claimed job and resolve its queue row.

    The classification commits in its own transaction before the job is marked
    done. If the process dies in between, the stuck-job sweep requeues the job
    and the idempotency guard makes the second run a no-op.
    """
    settings = settings or get_settings()
    try:
        async with sessions() as session:
            try:
                outcome = await run_message(
                    session, job.message_id, client=client, settings=settings
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    except LLMSchemaError as exc:
        await _dead_letter(sessions, job, f"schema failure: {exc}", raw=exc.raw)
        return RunOutcome(status="dead", message_id=job.message_id, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - any failure is a queue decision
        log.exception("job %s failed", job.id)
        async with sessions() as session:
            status = await queue.fail(session, job.id, repr(exc))
            await session.commit()
        return RunOutcome(
            status="dead" if status == "dead" else "failed",
            message_id=job.message_id,
            error=repr(exc),
        )

    async with sessions() as session:
        await queue.complete(session, job.id)
        await session.commit()
    return outcome


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _dead_letter(
    sessions: async_sessionmaker[AsyncSession], job: Job, error: str, *, raw: str
) -> None:
    """A schema failure has already been retried inside the client. One more
    round trip would only produce the same output, so it goes straight to
    dead, with the raw text on the job row for the human who reads it."""
    async with sessions() as session:
        await queue.fail(session, job.id, f"{error}\n--- raw ---\n{raw[:2000]}", max_attempts=1)
        await session.commit()


async def _persist_copy(
    session: AsyncSession, message_id: int, prior: Classification, settings: Settings
) -> Classification:
    """Copy a prior judgment onto this message under source='dedup'.

    The copy is a full row, not a pointer: the append-only history has to stay
    readable without joining through fingerprints that may later be pruned.
    """
    payload = dict(prior.payload)
    payload["deduped_from_message_id"] = prior.message_id
    return await repo.insert_classification(
        session,
        ClassificationIn(
            message_id=message_id,
            is_important=prior.is_important,
            category=prior.category,
            payload=payload,
            raw_response=prior.raw_response,
            model_id=settings.vllm_model_id,
            prompt_version=settings.prompt_version,
            schema_version=settings.schema_version,
            source="dedup",
            input_tokens=None,
            latency_ms=0,
        ),
    )


async def _embed_message(message, settings: Settings) -> list[float] | None:
    try:
        vectors = await embed(
            [embedding_text(message.subject, message.body_clean)], settings
        )
        return vectors[0] if vectors else None
    except Exception as exc:  # noqa: BLE001
        # An embedding failure must not cost the classification. Few-shot and
        # neighbour search degrade; triage does not.
        log.warning("embedding failed for message %s: %s", message.id, exc)
        return None


async def _post_actions(
    session: AsyncSession,
    message,
    *,
    fingerprint: str | None,
    embedding: list[float] | None = None,
    settings: Settings,
) -> None:
    if embedding is None:
        embedding = await _embed_message(message, settings)
    await repo.set_message_derived(
        session, message.id, fingerprint=fingerprint, embedding=embedding
    )


def known_categories() -> list[str]:
    """Re-exported so callers never build their own list."""
    return [c.value for c in Category]
