"""Registry policy and argument coercion. No database, no vLLM."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.fakes import FakeSession

from triage.agent.schema import StepArgs, ToolName
from triage.agent.tools import DEFAULT_LIMIT, REGISTRY, coerce_args


def test_only_the_two_queue_deletions_are_destructive():
    """The confirmation policy, as an executable statement.

    Adding a third destructive tool should be a deliberate act that fails this
    test, not something that slips in behind a default.
    """
    destructive = {s.name.value for s in REGISTRY.values() if s.destructive}
    assert destructive == {"clear_queue", "dequeue_message"}


def test_destructive_tools_explain_themselves():
    for spec in REGISTRY.values():
        if spec.destructive:
            assert spec.confirm_prompt, f"{spec.name.value} has no confirmation text"
            # The banner has to state the consequence, not just ask.
            assert "kept" in spec.confirm_prompt


def test_every_tool_is_described_and_bounded():
    for spec in REGISTRY.values():
        assert spec.description.strip()
        assert len(spec.description) < 250
        assert spec.timeout_s > 0


def test_stray_arguments_are_ignored_not_rejected():
    """The rule that keeps a wrong guess from costing an iteration.

    An 8B model will put `limit` on `read_message`. Failing that call teaches it
    nothing and burns one of six steps; dropping the field answers the question.
    """
    args = coerce_args(REGISTRY[ToolName.READ_MESSAGE], StepArgs(message_id=5, limit=10))
    assert args.message_id == 5
    assert not hasattr(args, "limit")


def test_a_genuinely_missing_argument_still_fails():
    with pytest.raises(ValidationError):
        coerce_args(REGISTRY[ToolName.READ_MESSAGE], StepArgs(limit=10))


def test_answer_requires_text():
    with pytest.raises(ValidationError):
        coerce_args(REGISTRY[ToolName.ANSWER], StepArgs())


def test_omitted_limit_falls_back_to_the_documented_default():
    args = coerce_args(REGISTRY[ToolName.LIST_UNPROCESSED], StepArgs())
    assert args.limit == DEFAULT_LIMIT


async def test_run_queue_limits_both_the_enqueue_and_the_run(monkeypatch):
    """The subtlety that makes "classify the newest 20" mean what it says.

    Passing the limit only to the runner would leave every other unclassified
    message sitting in the queue afterwards: the user asked for twenty and the
    queue depth would read four thousand.
    """
    seen = {}

    async def fake_enqueue(session, limit=10_000):
        seen["enqueued"] = limit
        return 20

    class FakeRunner:
        def start(self, *, limit=None):
            seen["ran"] = limit
            return True

        def status(self):  # pragma: no cover
            return {"processed": 0}

    monkeypatch.setattr("triage.db.repo.enqueue_unprocessed", fake_enqueue)
    monkeypatch.setattr("triage.api.tasks.runner", FakeRunner())

    spec = REGISTRY[ToolName.RUN_QUEUE]
    result = await spec.run(FakeSession(), coerce_args(spec, StepArgs(limit=20)))

    assert seen == {"enqueued": 20, "ran": 20}
    assert "20" in result.text


async def test_run_queue_reports_an_existing_drain_as_success(monkeypatch):
    """"Already running" is news, not an error.

    Surfacing it as a failure would make the model apologise for a queue that
    is working correctly, and would spend a strike doing it.
    """

    async def fake_enqueue(session, limit=10_000):
        return 3

    class BusyRunner:
        def start(self, *, limit=None):
            return False

        def status(self):
            return {"processed": 23}

    monkeypatch.setattr("triage.db.repo.enqueue_unprocessed", fake_enqueue)
    monkeypatch.setattr("triage.api.tasks.runner", BusyRunner())

    spec = REGISTRY[ToolName.RUN_QUEUE]
    result = await spec.run(FakeSession(), coerce_args(spec, StepArgs()))

    assert "already running" in result.text
    assert "23" in result.text
    assert result.meta["started"] is False


async def test_stop_queue_on_an_idle_queue_says_so(monkeypatch):
    class IdleRunner:
        running = False

    monkeypatch.setattr("triage.api.tasks.runner", IdleRunner())
    spec = REGISTRY[ToolName.STOP_QUEUE]
    result = await spec.run(FakeSession(), coerce_args(spec, StepArgs()))
    assert "nothing to stop" in result.text
    assert result.meta["stopped"] is False
