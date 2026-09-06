"""Assembling what the model actually reads.

The tool catalogue is rendered here rather than shipped inside the JSON schema,
because the schema goes to the grammar engine and the model never sees it. The
grammar decides what the model *can* say; this text decides what it *should*.
"""

from __future__ import annotations

from datetime import datetime

from triage.agent.budget import (
    OMITTED_MARKER,
    cap_result,
    cap_user_message,
    fit_scratchpad,
    trim_history,
)
from triage.agent.schema import ToolName
from triage.agent.tools import REGISTRY

SYSTEM = """You are the assistant inside a personal email triage system, \
talking to {owner}. Today is {today}.

The system stores emails in a database and classifies each one with a local \
model, giving it a category and an importance flag. Classification work goes \
through a queue.

Rules:
- Every reply is one JSON object: a short `thought`, a `tool`, and its `args`.
- Call one tool at a time and read its result before deciding what to do next.
- Never invent an email, a count, a sender or a date. If you have not seen it \
in a tool result, you do not know it.
- Use `answer` as soon as you can answer, and always as your last step. Put \
the whole reply to the user in `answer`'s `text`.
- Write `answer` as plain text with real line breaks - one item per \
line for a list. Do not use markdown: asterisks and hashes are shown \
to the user exactly as you type them.
- You have at most {max_iterations} tool calls before you must answer.
- Ids come from `find_messages` or `list_unprocessed`. Do not guess one.
- Do NOT start classifying unless the user asked you to. Queueing an \
email and classifying it are different requests: `run_queue` starts a run \
that can take hours, so only call it when the user actually asked to \
classify, process or run something.
- `clear_queue` and `dequeue_message` ask the user to confirm before they run.

Tools:
{catalogue}"""


def render_catalogue() -> str:
    """One line per tool: name, arguments, description.

    Arguments are listed from each tool's own model, not from ``StepArgs``, so
    the model is told what ``read_message`` takes rather than the union of what
    everything takes.
    """
    lines = []
    for name in ToolName:
        spec = REGISTRY[name]
        args = ", ".join(spec.args_model.model_fields) or "no arguments"
        lines.append(f"- {name.value}({args}): {spec.description}")
    return "\n".join(lines)


def render_system(owner: str, *, max_iterations: int, now: datetime | None = None) -> str:
    return SYSTEM.format(
        owner=owner,
        today=(now or datetime.now().astimezone()).strftime("%A %d %B %Y"),
        max_iterations=max_iterations,
        catalogue=render_catalogue(),
    )


def render_call(tool: str, args: dict) -> str:
    """How a step the model already took is echoed back to it.

    Rendered as the JSON it emitted, so its own prior turns stay in
    distribution: the model reads back exactly the shape it is being asked to
    produce.
    """
    import json

    return json.dumps({"tool": tool, "args": args}, separators=(",", ":"))


def build_messages(
    *,
    owner: str,
    history: list[list[dict[str, str]]],
    user_message: str,
    scratchpad: list[tuple[str, str]],
    max_iterations: int,
    nudge: str | None = None,
    now: datetime | None = None,
) -> tuple[list[dict[str, str]], bool]:
    """The full message list for one step, trimmed to fit.

    ``history`` is prior turns, each a list of ``{role, content}``; tool rows
    are already excluded by the store, which is the single biggest saving in
    the whole budget. ``scratchpad`` is this turn's completed steps as
    ``(call, result)`` pairs. Returns the messages and whether the user's own
    message had to be shortened.
    """
    messages = [
        {
            "role": "system",
            "content": render_system(owner, max_iterations=max_iterations, now=now),
        }
    ]

    prior, dropped = trim_history(history)
    if dropped:
        messages.append({"role": "user", "content": OMITTED_MARKER})
    messages.extend(prior)

    capped_user, shortened = cap_user_message(user_message)
    messages.append({"role": "user", "content": capped_user})

    for call, result in fit_scratchpad([(c, cap_result(r)) for c, r in scratchpad]):
        messages.append({"role": "assistant", "content": call})
        messages.append({"role": "user", "content": result})

    if nudge:
        messages.append({"role": "user", "content": nudge})
    return messages, shortened


def estimate_messages(messages: list[dict[str, str]]) -> int:
    from triage.agent.budget import estimate_tokens

    return sum(estimate_tokens(m["content"]) for m in messages)
