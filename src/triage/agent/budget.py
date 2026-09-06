"""Fitting a conversation into 8,192 tokens.

``--max-model-len 8192`` on this deployment is prompt *and* output together, and
that number is not a preference: it was chosen against a 0.55 GPU memory
fraction on an iGPU whose "GPU memory" is really system RAM behind a WSL2 VM.
Raising it costs KV cache the classification pipeline needs. So the agent lives
inside it.

The allocation:

    system prompt              470
    tool catalogue             850
    prior-turn history       1,400
    current-turn scratchpad  3,200
    current user message       500
    -----------------------------
    prompt                   6,420
    output reserve             768
    estimator headroom       1,004
    -----------------------------
                             8,192

The headroom is real slack, not padding for a rainy day. There is no tokenizer
here (see ``estimate_tokens``), so the budget is enforced against an estimate,
and an estimate that is occasionally wrong in the wrong direction produces a
400 from vLLM instead of a slightly long prompt.
"""

from __future__ import annotations

from math import ceil

#: Deliberately pessimistic. Qwen3 on English prose is nearer 3.8 characters per
#: token, so this over-counts by roughly 25%. That is the entire safety margin
#: and it is free: the cost of over-estimating is dropping one old turn early,
#: and the cost of under-estimating is a failed request mid-conversation.
CHARS_PER_TOKEN = 3.0

SYSTEM_BUDGET = 470
CATALOGUE_BUDGET = 850
HISTORY_BUDGET = 1_400
SCRATCHPAD_BUDGET = 3_200
USER_MESSAGE_BUDGET = 500
PROMPT_BUDGET = (
    SYSTEM_BUDGET + CATALOGUE_BUDGET + HISTORY_BUDGET + SCRATCHPAD_BUDGET + USER_MESSAGE_BUDGET
)

#: Per tool result, before the whole scratchpad is considered.
RESULT_BUDGET = 700

TRUNCATED_MARKER = "\n... (truncated - ask for fewer rows or a narrower query)"
OMITTED_MARKER = "[earlier messages in this conversation were omitted]"
SHORTENED_NOTE = "\n[your message was shortened to fit the model's context]"


def estimate_tokens(text: str) -> int:
    """Characters over a constant, plus a few for turn framing.

    Why not the real tokenizer: loading Qwen3's means downloading it from
    HuggingFace, and the whole point of this box is that it makes no outbound
    requests -- the mailbox is on it. ``transformers`` is installed (via
    sentence-transformers) but the vocabulary is not, and shipping one to save a
    division is not worth it.

    The estimate is checked against reality on every step: ``_call`` returns
    ``usage.prompt_tokens``, and the loop logs estimate-versus-actual at DEBUG.
    If the ratio drifts, this one constant is the thing to change.
    """
    return ceil(len(text) / CHARS_PER_TOKEN) + 4


def cap_result(text: str, budget: int = RESULT_BUDGET) -> str:
    """Trim one tool result, on a line boundary, with an instructive marker.

    The marker tells the model what to *do* about it, because ``limit`` is a
    field it can set. "Truncated" alone would just be a fact it cannot act on.
    """
    if estimate_tokens(text) <= budget:
        return text
    keep = int(budget * CHARS_PER_TOKEN)
    cut = text[:keep]
    # Prefer a whole line: half a table row reads as corrupt data rather than
    # as a shortened list.
    newline = cut.rfind("\n")
    if newline > keep // 2:
        cut = cut[:newline]
    return cut.rstrip() + TRUNCATED_MARKER


def cap_user_message(text: str, budget: int = USER_MESSAGE_BUDGET) -> tuple[str, bool]:
    """Trim the user's own message, reporting whether it had to.

    Someone pastes a log file. The full text is still stored on the row, so the
    transcript stays honest; only the model's copy is shortened, and the page
    says so rather than silently dropping half the question.
    """
    if estimate_tokens(text) <= budget:
        return text, False
    return text[: int(budget * CHARS_PER_TOKEN)].rstrip() + SHORTENED_NOTE, True


def trim_history(
    turns: list[list[dict[str, str]]], budget: int = HISTORY_BUDGET
) -> tuple[list[dict[str, str]], bool]:
    """Keep the newest whole turns that fit.

    Whole turns, never half of one: a user question without its answer, or an
    answer with no question, is worse than no history at all -- the model
    invents the missing half. Returns the flattened messages and whether
    anything was dropped, so the caller can prepend the omission marker.
    """
    kept: list[list[dict[str, str]]] = []
    used = 0
    for turn in reversed(turns):
        cost = sum(estimate_tokens(m["content"]) for m in turn)
        if used + cost > budget:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    dropped = len(kept) < len(turns)
    return [m for turn in kept for m in turn], dropped


def fit_scratchpad(
    entries: list[tuple[str, str]], budget: int = SCRATCHPAD_BUDGET
) -> list[tuple[str, str]]:
    """Elide the oldest results until the current turn's scratchpad fits.

    Entries are ``(call_line, result_text)``. The call line always survives --
    that is what stops the model calling the same tool a second time because it
    has forgotten it already did. Only the body is replaced.
    """
    out = list(entries)

    def total() -> int:
        return sum(estimate_tokens(c) + estimate_tokens(r) for c, r in out)

    for i in range(len(out)):
        if total() <= budget:
            break
        call, _ = out[i]
        out[i] = (call, f"[result of step {i + 1} elided to save room]")
    return out
