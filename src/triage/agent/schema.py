"""The one object the model is allowed to emit, and the grammar built from it.

Every agent step is a single JSON object validated against ``AgentStep``. This
is the same mechanism classification uses -- ``structured_outputs.json`` on the
request, a Pydantic model on the way back -- and it is deliberate: this vLLM
server runs without ``--enable-auto-tool-choice``, so native tool calling is
not available, and the guided-JSON path is the one that is already proven on
this hardware.

The schema is kept *flat* on purpose. ``args`` is one fixed-key object whose
every field is optional, rather than a discriminated union per tool. A union
would be a truer type and a worse grammar: deep ``anyOf`` is where constrained
decoding gets slow and where an 8B model gets confused about which branch it is
in. Per-tool argument validation still happens -- in Python, in ``tools.py``,
where a failure can be turned into a sentence the model can act on.
"""

from __future__ import annotations

import copy
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from triage.taxonomy import Category


class ToolName(StrEnum):
    """Every tool the model can name. The grammar admits nothing else."""

    FIND_MESSAGES = "find_messages"
    READ_MESSAGE = "read_message"
    LIST_UNPROCESSED = "list_unprocessed"
    LIST_REVIEW_QUEUE = "list_review_queue"
    MAILBOX_STATS = "mailbox_stats"
    QUEUE_STATUS = "queue_status"
    SYNC_MAIL = "sync_mail"
    ENQUEUE_UNPROCESSED = "enqueue_unprocessed"
    ENQUEUE_MESSAGE = "enqueue_message"
    RUN_QUEUE = "run_queue"
    STOP_QUEUE = "stop_queue"
    RETRY_DEAD_JOBS = "retry_dead_jobs"
    CLEAR_QUEUE = "clear_queue"
    DEQUEUE_MESSAGE = "dequeue_message"
    ANSWER = "answer"


class StepArgs(BaseModel):
    """The whole argument surface of every tool, in seven optional fields.

    Seven, not seventy: the flat schema only works because the tools were
    designed to share argument names. ``limit`` means the same thing to
    ``find_messages`` and ``retry_dead_jobs``; ``days`` means the same thing to
    ``find_messages`` and ``sync_mail``. Adding an eighth field is allowed;
    adding a tool that needs an argument no other tool has is the thing to
    resist, and there is a test that fails when someone does it anyway.

    Bounds are here rather than in the executors so the *grammar* enforces
    them. A model that cannot emit ``limit: 500`` never wastes a step being
    told it was too big.
    """

    model_config = ConfigDict(extra="forbid")

    message_id: int | None = Field(default=None, ge=1)
    query: str | None = Field(default=None, max_length=120)
    category: Category | None = None
    important_only: bool | None = None
    limit: int | None = Field(default=None, ge=1, le=25)
    days: int | None = Field(default=None, ge=1, le=90)
    text: str | None = Field(default=None, max_length=2000)


class AgentStep(BaseModel):
    """One decision: a caption, a tool, and its arguments."""

    model_config = ConfigDict(extra="forbid")

    #: Not chain-of-thought -- the grammar forbids a <think> block, and
    #: enable_thinking is off for the same reason it is off in classification.
    #: This is a progress caption, shown above the tool chip in the transcript,
    #: which is what a user staring at a 60-second turn actually needs. Capped
    #: at the same 200 characters as Triage.reason, which is a proven length.
    thought: str = Field(max_length=200)
    tool: ToolName
    args: StepArgs = Field(default_factory=StepArgs)


def _strip_annotations(node: Any) -> Any:
    """Drop ``title`` and ``description`` from a schema, in place.

    These are documentation, and nothing downstream reads them: the schema goes
    into ``extra_body`` where vLLM compiles it to a grammar, and the model never
    sees it. Left in, every field docstring in this module would ride along on
    every single agent request. The tool descriptions the model *does* read are
    rendered separately, in prompt.py.
    """
    if isinstance(node, dict):
        node.pop("title", None)
        node.pop("description", None)
        for value in node.values():
            _strip_annotations(value)
    elif isinstance(node, list):
        for value in node:
            _strip_annotations(value)
    return node


def agent_step_json_schema(only: list[ToolName] | None = None) -> dict[str, Any]:
    """The JSON schema for one step, optionally restricted to certain tools.

    ``only`` is the loop's forcing mechanism, and it is the single highest
    leverage thing in the agent. When the model has used its last iteration, or
    has made three invalid calls in a row, or has run out of context, the fix
    is not to *ask* it to stop -- an 8B that has already made two bad calls
    will cheerfully make a third. Narrowing the enum to ``[answer]`` makes
    stopping the only sequence of tokens the grammar permits.

    Pydantic emits the enum in ``$defs``, so this rewrites it there on a deep
    copy; the returned schema is otherwise identical to the unrestricted one.
    """
    schema = _strip_annotations(copy.deepcopy(AgentStep.model_json_schema()))
    if only is None:
        return schema
    if not only:
        raise ValueError("agent_step_json_schema(only=[]) would admit no tool at all")
    values = [t.value for t in only]
    for name, defn in schema.get("$defs", {}).items():
        if name == ToolName.__name__:
            defn["enum"] = values
            break
    else:  # pragma: no cover - only reachable if pydantic stops using $defs
        raise RuntimeError(f"{ToolName.__name__} not found in $defs; schema shape changed")
    return schema
