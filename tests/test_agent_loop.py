"""The turn state machine, driven by a scripted model. No DB, no vLLM.

One case per row of the error table in the design. The two that matter most are
the forcing tests: after six iterations, and after three strikes, the *grammar*
handed to the model must admit only `answer`. Asking a small model to stop does
not reliably stop it; making "answer" the only reachable token does.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.fakes import FakeChatStore, ScriptedLLM, answer, fake_sessionmaker, step

from triage.agent import errors as err
from triage.agent import loop as agent_loop
from triage.agent.loop import MAX_ERROR_STRIKES, MAX_ITERATIONS, run_turn
from triage.agent.schema import ToolName
from triage.llm.client import LLMBadRequest, LLMUnavailable


@pytest.fixture
def store(monkeypatch):
    fake = FakeChatStore()
    monkeypatch.setattr(agent_loop.chat, "add_message", fake.add_message)
    monkeypatch.setattr(agent_loop.chat, "finish_message", fake.finish_message)
    monkeypatch.setattr(agent_loop.chat, "history_for_model", fake.history_for_model)
    monkeypatch.setattr(agent_loop.chat, "touch_thread", fake.touch_thread)
    return fake


async def approve(_message_id: int) -> bool:
    return True


async def decline(_message_id: int) -> bool:
    return False


async def drive(client, store, *, confirm=approve, user_message="what is queued?"):
    sessions, _ = fake_sessionmaker()
    await run_turn(
        sessions,
        client,
        thread_id=1,
        turn_id=1,
        user_message=user_message,
        owner="me@example.com",
        confirm=confirm,
    )
    return store


def stub_tool(monkeypatch, name: ToolName, fn, **overrides):
    """Swap one tool's executor, keeping the rest of its spec."""
    import dataclasses

    from triage.agent.tools import REGISTRY

    spec = dataclasses.replace(REGISTRY[name], run=fn, **overrides)
    monkeypatch.setitem(REGISTRY, name, spec)
    return spec


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


async def test_tool_then_answer_writes_rows_in_order(store, monkeypatch):
    from triage.agent.tools import ToolResult

    async def ok(session, args):
        return ToolResult(text="pending 3", summary="pending 3")

    stub_tool(monkeypatch, ToolName.QUEUE_STATUS, ok)
    client = ScriptedLLM([step(ToolName.QUEUE_STATUS), answer("You have 3 queued.")])

    await drive(client, store)

    assert store.roles() == ["tool", "assistant"]
    assert store.by_role("tool")[0]["status"] == "done"
    assert store.final()["content"] == "You have 3 queued."


async def test_the_tool_row_is_written_before_the_tool_runs(store, monkeypatch):
    """What makes the two-second poll show progress instead of a spinner."""
    from triage.agent.tools import ToolResult

    seen_during_run = []

    async def slow(session, args):
        seen_during_run.append(list(store.roles()))
        return ToolResult(text="ok", summary="ok")

    stub_tool(monkeypatch, ToolName.QUEUE_STATUS, slow)
    client = ScriptedLLM([step(ToolName.QUEUE_STATUS), answer()])

    await drive(client, store)
    assert seen_during_run == [["tool"]]


async def test_tool_result_is_fed_back_to_the_model(store, monkeypatch):
    from triage.agent.tools import ToolResult

    async def ok(session, args):
        return ToolResult(text="pending 41", summary="pending 41")

    stub_tool(monkeypatch, ToolName.QUEUE_STATUS, ok)
    client = ScriptedLLM([step(ToolName.QUEUE_STATUS), answer()])

    await drive(client, store)

    second_call = "\n".join(m["content"] for m in client.calls[1])
    assert "TOOL RESULT queue_status" in second_call
    assert "pending 41" in second_call


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


async def test_bad_arguments_are_explained_and_the_loop_continues(store):
    client = ScriptedLLM([step(ToolName.READ_MESSAGE), answer("I need an id first.")])

    await drive(client, store)

    feedback = "\n".join(m["content"] for m in client.calls[1])
    assert "bad_args" in feedback
    assert "message_id" in feedback
    assert store.final()["status"] == "done"


async def test_a_raising_tool_is_reported_without_a_traceback(store, monkeypatch):
    async def boom(session, args):
        raise RuntimeError("Google rejected the stored credentials")

    stub_tool(monkeypatch, ToolName.SYNC_MAIL, boom)
    client = ScriptedLLM([step(ToolName.SYNC_MAIL), answer("Sync failed.")])

    await drive(client, store)

    feedback = "\n".join(m["content"] for m in client.calls[1])
    assert "Google rejected the stored credentials" in feedback
    # No class names, no repr: the model imitates what it is shown.
    assert "RuntimeError" not in feedback
    assert "Traceback" not in feedback
    assert store.by_role("tool")[0]["status"] == "error"


async def test_a_slow_tool_times_out_and_the_model_is_told_where_to_look(store, monkeypatch):
    async def hangs(session, args):
        await asyncio.sleep(5)

    stub_tool(monkeypatch, ToolName.SYNC_MAIL, hangs, timeout_s=0.01)
    client = ScriptedLLM([step(ToolName.SYNC_MAIL), answer("Still going.")])

    await drive(client, store)

    feedback = "\n".join(m["content"] for m in client.calls[1])
    assert "timeout" in feedback
    assert "queue_status" in feedback


async def test_a_missing_message_is_not_a_strike(store):
    """Discovering that id 9999 does not exist is a finding, not a mistake.

    Counting it would spend a third of the strike budget on the model doing
    exactly the right thing.
    """
    replies = [step(ToolName.READ_MESSAGE, message_id=9990 + i) for i in range(4)]
    client = ScriptedLLM([*replies, answer("None of those exist.")])

    async def missing(session, args):
        from triage.agent.errors import ToolErrorCode, ToolFailure

        raise ToolFailure(ToolErrorCode.NOT_FOUND, "there is no email with that id.")

    import dataclasses

    from triage.agent.tools import REGISTRY

    original = REGISTRY[ToolName.READ_MESSAGE]
    REGISTRY[ToolName.READ_MESSAGE] = dataclasses.replace(original, run=missing)
    try:
        await drive(client, store)
    finally:
        REGISTRY[ToolName.READ_MESSAGE] = original

    # Four not_founds did not force an early answer.
    assert client.tools_offered(3) != ["answer"]
    assert store.final()["content"] == "None of those exist."


# ---------------------------------------------------------------------------
# forcing
# ---------------------------------------------------------------------------


async def test_the_grammar_is_narrowed_after_the_iteration_cap(store, monkeypatch):
    from triage.agent.tools import ToolResult

    async def ok(session, args):
        return ToolResult(text="ok", summary="ok")

    stub_tool(monkeypatch, ToolName.QUEUE_STATUS, ok)
    client = ScriptedLLM([*[step(ToolName.QUEUE_STATUS)] * MAX_ITERATIONS, answer("Enough.")])

    await drive(client, store)

    for i in range(MAX_ITERATIONS):
        assert len(client.tools_offered(i)) == len(ToolName)
    # The final call can only produce an answer.
    assert client.tools_offered(MAX_ITERATIONS) == ["answer"]
    assert store.final()["content"] == "Enough."


async def test_the_grammar_is_narrowed_after_three_strikes(store):
    client = ScriptedLLM(
        [*[step(ToolName.READ_MESSAGE)] * MAX_ERROR_STRIKES, answer("I could not do that.")]
    )

    await drive(client, store)

    assert client.tools_offered(MAX_ERROR_STRIKES) == ["answer"]
    assert store.final()["content"] == "I could not do that."


# ---------------------------------------------------------------------------
# confirmation
# ---------------------------------------------------------------------------


async def test_an_approved_destructive_tool_runs(store, monkeypatch):
    from triage.agent.tools import ToolResult

    ran = []

    async def clear(session, args):
        ran.append(True)
        return ToolResult(text="deleted 12 jobs", summary="cleared 12")

    stub_tool(monkeypatch, ToolName.CLEAR_QUEUE, clear)
    client = ScriptedLLM([step(ToolName.CLEAR_QUEUE), answer("Cleared.")])

    await drive(client, store, confirm=approve)

    assert ran == [True]
    row = store.by_role("tool")[0]
    # It was written as awaiting_confirm first -- that row is what the confirm
    # POST targets and what a restart sweeps.
    assert row["status"] == "done"


async def test_a_declined_destructive_tool_does_not_run_and_ends_the_turn(store, monkeypatch):
    ran = []

    async def clear(session, args):  # pragma: no cover - must not be called
        ran.append(True)

    stub_tool(monkeypatch, ToolName.CLEAR_QUEUE, clear)
    client = ScriptedLLM([step(ToolName.CLEAR_QUEUE), answer("Cancelled, nothing was deleted.")])

    await drive(client, store, confirm=decline)

    assert ran == []
    assert store.by_role("tool")[0]["status"] == "declined"
    feedback = "\n".join(m["content"] for m in client.calls[1])
    assert "declined" in feedback
    # Nothing left to explore after a no.
    assert client.tools_offered(1) == ["answer"]


# ---------------------------------------------------------------------------
# aborts -- every one must still write a terminal row
# ---------------------------------------------------------------------------


async def test_two_unparseable_replies_abort_the_turn(store):
    client = ScriptedLLM(["not json at all", "{still not"])

    await drive(client, store)

    final = store.final()
    assert final["status"] == "error"
    assert final["content"] == err.SCHEMA_FAILED


async def test_a_truncated_reply_is_treated_as_a_schema_failure(store):
    """It *is* a truncated object, not a transport problem.

    _call appends this marker on finish_reason=length, and the raw text is kept
    so the failure is inspectable.
    """
    cut = '{"thought":"x","tool":"answer","args":{"text":"aaa' + agent_loop.TRUNCATION_MARKER
    client = ScriptedLLM([cut, cut])

    await drive(client, store)
    assert store.final()["content"] == err.SCHEMA_FAILED


async def test_an_unreachable_model_names_the_url(store):
    client = ScriptedLLM([LLMUnavailable("connection refused")])

    await drive(client, store)

    final = store.final()
    assert final["status"] == "error"
    assert "127.0.0.1:8000" in final["content"]


async def test_an_over_long_prompt_shrinks_and_retries_before_giving_up(store):
    """A blown context must not read as "the server is down".

    LLMBadRequest is a subclass of LLMUnavailable so classification is
    unaffected, but the agent treats it completely differently: halve the
    context and try once more.
    """
    client = ScriptedLLM([LLMBadRequest("maximum context length"), answer("Recovered.")])

    await drive(client, store)

    assert len(client.calls) == 2
    assert store.final()["content"] == "Recovered."


async def test_a_persistently_over_long_prompt_says_to_start_a_new_chat(store):
    client = ScriptedLLM([LLMBadRequest("too long"), LLMBadRequest("still too long")])

    await drive(client, store)

    assert store.final()["content"] == err.CONTEXT_FULL


async def test_an_empty_answer_still_produces_something_to_read(store):
    """A blank assistant bubble looks like a crash. Say something instead."""
    client = ScriptedLLM([step(ToolName.ANSWER, text="   ")])

    await drive(client, store)
    assert store.final()["content"].strip()
