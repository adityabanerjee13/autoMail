"""Transport models between the database layer and everything above it.

``db/repo.py`` returns these, never SQLAlchemy rows. Keeping ORM objects out of
worker and API code is what prevents lazy-loading and detached-instance errors
from showing up inside the pipeline.

The LLM output contract lives in ``llm/schemas.py``; this module is about rows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["pending", "running", "done", "dead"]
ClassificationSource = Literal["llm", "dedup", "human"]


class MessageIn(BaseModel):
    """A parsed message ready to be written. No database identity yet."""

    gmail_id: str
    thread_id: str
    message_id_hdr: str | None = None
    from_addr: str
    to_addrs: list[str] = Field(default_factory=list)
    cc_addrs: list[str] = Field(default_factory=list)
    subject: str | None = None
    body_clean: str = ""
    snippet: str | None = None
    labels: list[str] = Field(default_factory=list)
    headers: dict[str, Any] = Field(default_factory=dict)
    internal_date: datetime


class Message(MessageIn):
    """A persisted message row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    fingerprint: str | None = None
    created_at: datetime


class Classification(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    message_id: int
    is_important: bool
    category: str
    payload: dict[str, Any]
    raw_response: str
    model_id: str
    prompt_version: str
    schema_version: str
    source: ClassificationSource
    input_tokens: int | None = None
    latency_ms: int | None = None
    created_at: datetime


class ClassificationIn(BaseModel):
    """An append to ``classifications``. There is deliberately no update model."""

    message_id: int
    is_important: bool
    category: str
    payload: dict[str, Any]
    raw_response: str
    model_id: str
    prompt_version: str
    schema_version: str
    source: ClassificationSource
    input_tokens: int | None = None
    latency_ms: int | None = None


class Job(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    message_id: int
    status: JobStatus
    attempts: int
    run_after: datetime
    last_error: str | None = None
    started_at: datetime | None = None
    created_at: datetime


class SyncState(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = 1
    history_id: str | None = None
    watch_expiry: datetime | None = None
    last_reconcile_at: datetime | None = None


class MessageWithClassification(BaseModel):
    """Join used by the review UI: a message plus its latest judgment."""

    message: Message
    classification: Classification | None = None
    needs_review: bool = False
