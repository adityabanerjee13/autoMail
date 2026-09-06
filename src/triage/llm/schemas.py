"""The model's output contract.

One call per message: classification and extraction share this single schema.
Two calls would double GPU time for no added information.

``Triage.model_json_schema()`` is what goes to vLLM as the structured-outputs
schema. Nothing in this repository parses free text out of a completion.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from triage.taxonomy import Category

Confidence = Literal["low", "medium", "high"]


class Triage(BaseModel):
    is_important: bool
    importance_confidence: Confidence
    category: Category
    category_confidence: Confidence
    action_required: bool
    deadline: date | None = None
    # Read by a human when a classification looks wrong, and mined for
    # category definitions that overlap. It is not decorative -- step 4 of the
    # build order is reading 50 of these by hand before the backfill.
    reason: str = Field(max_length=200)

    @property
    def needs_review(self) -> bool:
        """Low confidence on either axis routes the message to review.

        Coarse buckets rather than a float: LLMs produce badly calibrated
        numbers but usable buckets.
        """
        return "low" in (self.importance_confidence, self.category_confidence)


def triage_json_schema() -> dict[str, Any]:
    """The JSON Schema handed to vLLM's structured-outputs decoder."""
    return Triage.model_json_schema()


def example_payload() -> dict[str, Any]:
    """A filled example, rendered into the prompt so the model sees the shape.

    Derived from the model rather than typed out again, so it cannot drift from
    the schema the decoder is actually enforcing.
    """
    return Triage(
        is_important=True,
        importance_confidence="high",
        category=Category.FINANCE,
        category_confidence="high",
        action_required=True,
        deadline=date(2026, 3, 14),
        reason=(
            "Card statement with a payment due date; the recipient must act "
            "before the due date."
        ),
    ).model_dump(mode="json")
