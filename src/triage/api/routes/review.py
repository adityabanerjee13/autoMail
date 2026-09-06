"""The review queue and the correction endpoint.

A correction is an INSERT with source='human', never an UPDATE. The model's
original judgment stays on the record: the pair (what the model said, what the
human said) is the dataset phase 2 trains on, and deleting the wrong half
destroys it.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from triage.api.deps import get_session, templates
from triage.config import get_settings
from triage.db import repo
from triage.schemas import ClassificationIn
from triage.taxonomy import Category, category_values

router = APIRouter()


@router.get("/review", response_class=HTMLResponse)
async def review_queue(
    request: Request,
    page: int = 1,
    session: AsyncSession = Depends(get_session),
):
    limit = 50
    rows = await repo.review_queue(session, limit=limit, offset=(page - 1) * limit)
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "rows": rows,
            "categories": category_values(),
            "page": page,
            "has_next": len(rows) == limit,
        },
    )


@router.post("/review/{message_id}", response_class=HTMLResponse)
async def submit_correction(
    request: Request,
    message_id: int,
    category: str = Form(...),
    is_important: bool = Form(False),
    action_required: bool = Form(False),
    reason: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    settings = get_settings()

    if category not in set(category_values()):
        # The dropdown is generated from the taxonomy, so this only fires on a
        # hand-made request -- or on a taxonomy edit that left a stale page open.
        raise HTTPException(status_code=422, detail=f"unknown category: {category}")

    message = await repo.get_message(session, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="no such message")

    prior = await repo.latest_classification(session, message_id)
    payload = {
        "is_important": is_important,
        # A human is not guessing about their own mailbox.
        "importance_confidence": "high",
        "category": Category(category).value,
        "category_confidence": "high",
        "action_required": action_required,
        "deadline": (prior.payload.get("deadline") if prior else None),
        "reason": (reason or "human correction")[:200],
        "corrected_from": (
            {
                "classification_id": prior.id,
                "is_important": prior.is_important,
                "category": prior.category,
                "source": prior.source,
            }
            if prior
            else None
        ),
    }

    stored = await repo.insert_classification(
        session,
        ClassificationIn(
            message_id=message_id,
            is_important=is_important,
            category=payload["category"],
            payload=payload,
            # There is no model output here; keeping the field honest matters
            # more than keeping it non-empty.
            raw_response=json.dumps({"source": "human", "submitted_by": "review-ui"}),
            model_id=settings.vllm_model_id,
            prompt_version=settings.prompt_version,
            schema_version=settings.schema_version,
            source="human",
            input_tokens=None,
            latency_ms=None,
        ),
    )
    await session.commit()

    return templates.TemplateResponse(
        request,
        "_corrected.html",
        {"message": message, "classification": stored},
    )
