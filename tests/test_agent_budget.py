"""The token arithmetic that keeps a turn inside 8,192.

The load-bearing test here is ``test_worst_case_conversation_fits``. It is the
one that fails the day someone adds a sixteenth tool, or lets a result renderer
off its leash -- both of which would otherwise surface as an opaque 400 from
vLLM partway through a real conversation.
"""

from __future__ import annotations

from triage.agent.budget import (
    CHARS_PER_TOKEN,
    HISTORY_BUDGET,
    OMITTED_MARKER,
    PROMPT_BUDGET,
    RESULT_BUDGET,
    SHORTENED_NOTE,
    TRUNCATED_MARKER,
    cap_result,
    cap_user_message,
    estimate_tokens,
    fit_scratchpad,
    trim_history,
)
from triage.agent.loop import AGENT_MAX_OUTPUT_TOKENS
from triage.agent.prompt import build_messages, estimate_messages, render_catalogue

MAX_MODEL_LEN = 8192


def test_estimate_is_monotonic_and_pessimistic():
    """Over-counting is the entire safety margin, and it is free.

    Qwen3 on English is nearer 3.8 characters per token. Estimating at 3.0
    means the budget is enforced with roughly 25% of slack, so ordinary drift
    drops one old turn early instead of producing a rejected request.
    """
    assert CHARS_PER_TOKEN <= 3.5
    assert estimate_tokens("a" * 100) < estimate_tokens("a" * 200)
    prose = "The quick brown fox jumps over the lazy dog. " * 20
    assert estimate_tokens(prose) > len(prose.split())


def test_cap_result_truncates_on_a_line_boundary():
    """Half a table row reads as corrupt data; a short table reads as a short table."""
    table = "\n".join(f"{i} | 09-0{i % 9} | a@b.com | subject {i}" for i in range(400))
    out = cap_result(table)
    assert out.endswith(TRUNCATED_MARKER)
    body = out[: -len(TRUNCATED_MARKER)]
    assert estimate_tokens(out) <= RESULT_BUDGET * 1.2
    # Every retained line is whole.
    assert all(line.count("|") == 3 for line in body.strip().splitlines()[1:] if line)


def test_cap_result_marker_tells_the_model_what_to_do():
    """"Truncated" alone is a fact it cannot act on; `limit` is a field it can set."""
    assert "fewer rows" in TRUNCATED_MARKER or "narrower" in TRUNCATED_MARKER


def test_short_results_pass_through_untouched():
    assert cap_result("pending 3, done 5") == "pending 3, done 5"


def test_trim_history_keeps_newest_whole_turns():
    turns = [
        [
            {"role": "user", "content": f"q{i}" * 400},
            {"role": "assistant", "content": f"a{i}" * 400},
        ]
        for i in range(20)
    ]
    kept, dropped = trim_history(turns)
    assert dropped is True
    assert estimate_tokens("".join(m["content"] for m in kept)) <= HISTORY_BUDGET
    # Never half a turn: a question without its answer makes the model invent one.
    assert len(kept) % 2 == 0
    # And it kept the newest, not the oldest.
    assert "q19" in kept[-2]["content"]


def test_trim_history_leaves_a_short_thread_alone():
    turns = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]]
    kept, dropped = trim_history(turns)
    assert dropped is False
    assert len(kept) == 2


def test_dropped_history_is_announced():
    turns = [
        [{"role": "user", "content": "x" * 3000}, {"role": "assistant", "content": "y" * 3000}]
        for _ in range(5)
    ]
    messages, _ = build_messages(
        owner="a@b.com", history=turns, user_message="now what?", scratchpad=[], max_iterations=6
    )
    assert any(OMITTED_MARKER in m["content"] for m in messages)


def test_fit_scratchpad_elides_oldest_bodies_but_keeps_the_calls():
    """The call line surviving is what stops the model repeating a tool it already ran."""
    entries = [(f'{{"tool":"find_messages","args":{{"limit":{i}}}}}', "z" * 6000) for i in range(6)]
    out = fit_scratchpad(entries)
    assert [c for c, _ in out] == [c for c, _ in entries]
    assert "elided" in out[0][1]
    # Newest result survives intact -- it is the one the next decision needs.
    assert out[-1][1] == entries[-1][1]


def test_a_pasted_log_file_is_shortened_not_dropped():
    text, shortened = cap_user_message("x" * 20_000)
    assert shortened is True
    assert text.endswith(SHORTENED_NOTE)
    assert len(text) < 20_000


def test_normal_questions_are_untouched():
    text, shortened = cap_user_message("what is in my queue?")
    assert shortened is False
    assert text == "what is in my queue?"


def test_system_prompt_stays_within_its_slice():
    """The catalogue has its own budget, so measure the rules on their own.

    Added after adding two rules silently pushed the system prompt over its
    allocation twice; the worst-case test still passed both times, because the
    headroom absorbed it, so nothing said the documented number had gone stale.
    """
    from triage.agent.budget import SYSTEM_BUDGET
    from triage.agent.prompt import render_system

    rules = estimate_tokens(render_system("me@example.com", max_iterations=6))
    assert rules - estimate_tokens(render_catalogue()) <= SYSTEM_BUDGET


def test_catalogue_stays_within_its_slice():
    """Fails when a sixteenth tool is added, or a description grows a paragraph."""
    assert estimate_tokens(render_catalogue()) <= 850


def test_worst_case_conversation_fits():
    """Twenty turns of history plus six maximal tool results, end to end.

    This is the whole budget as a single assertion. The output reserve is
    included because --max-model-len is prompt *and* completion together.
    """
    history = [
        [{"role": "user", "content": "q" * 900}, {"role": "assistant", "content": "a" * 1800}]
        for _ in range(20)
    ]
    scratchpad = [('{"tool":"find_messages","args":{}}', "z" * 8000) for _ in range(6)]

    messages, _ = build_messages(
        owner="someone@example.com",
        history=history,
        user_message="q" * 4000,
        scratchpad=scratchpad,
        max_iterations=6,
    )
    prompt = estimate_messages(messages)
    assert prompt <= PROMPT_BUDGET, f"prompt {prompt} over budget {PROMPT_BUDGET}"
    assert prompt + AGENT_MAX_OUTPUT_TOKENS <= MAX_MODEL_LEN
