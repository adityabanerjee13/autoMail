"""The grammar contract.

Everything here is pure and fast, and it guards the invariants that make the
flat-schema design work. The one that matters most is
``test_every_tool_argument_exists_on_step_args``: the day someone adds a
parameter to one tool without adding it to StepArgs, the model becomes
physically unable to express that argument, and nothing else in the system
would notice.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from triage.agent.schema import AgentStep, StepArgs, ToolName, agent_step_json_schema
from triage.agent.tools import REGISTRY


def test_step_is_flat():
    schema = agent_step_json_schema()
    assert set(schema["properties"]) == {"thought", "tool", "args"}
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["StepArgs"]["additionalProperties"] is False


def test_schema_carries_no_documentation():
    """Titles and descriptions are dead weight on every request.

    The schema goes to the grammar engine; the model never reads it. Left in,
    every field docstring in schema.py would ride along on each generation.
    """
    blob = json.dumps(agent_step_json_schema())
    assert "title" not in blob
    assert "description" not in blob


def test_registry_covers_every_tool_name():
    assert set(REGISTRY) == set(ToolName)


def test_every_tool_name_is_reachable_in_the_grammar():
    enum = agent_step_json_schema()["$defs"]["ToolName"]["enum"]
    assert sorted(enum) == sorted(t.value for t in ToolName)


def test_every_tool_argument_exists_on_step_args():
    """The invariant the flat schema rests on.

    Each tool validates against its own model, but the model can only *emit*
    fields StepArgs declares. A tool argument missing from StepArgs is one the
    model can never send, and the failure is silent: the tool just always sees
    its default.
    """
    allowed = set(StepArgs.model_fields)
    for spec in REGISTRY.values():
        extra = set(spec.args_model.model_fields) - allowed
        assert not extra, f"{spec.name.value} declares {extra}, which StepArgs cannot express"


def test_narrowing_restricts_the_enum_and_nothing_else():
    full = agent_step_json_schema()
    narrow = agent_step_json_schema(only=[ToolName.ANSWER])
    assert narrow["$defs"]["ToolName"]["enum"] == ["answer"]
    # Everything but that one list must be identical.
    full["$defs"]["ToolName"]["enum"] = ["answer"]
    assert full == narrow


def test_narrowing_to_nothing_is_refused():
    """An empty enum compiles to a grammar that admits no tool at all.

    The model would then be unable to emit any valid object and the turn would
    dead-end in a schema error, which reads like a model failure and is not.
    """
    with pytest.raises(ValueError):
        agent_step_json_schema(only=[])


def test_valid_step_parses():
    step = AgentStep.model_validate_json(
        '{"thought":"check the queue","tool":"queue_status","args":{}}'
    )
    assert step.tool is ToolName.QUEUE_STATUS


def test_args_may_be_omitted_entirely():
    """Half the tools take no arguments; requiring `args: {}` wastes tokens."""
    step = AgentStep.model_validate_json('{"thought":"x","tool":"mailbox_stats"}')
    assert step.args.model_dump(exclude_none=True) == {}


@pytest.mark.parametrize(
    "raw",
    [
        '{"thought":"x","tool":"nope","args":{}}',
        '{"thought":"x","tool":"answer","args":{"nonsense":1}}',
        '{"thought":"x","tool":"answer","args":{},"extra":1}',
        '{"tool":"answer","args":{}}',
    ],
)
def test_malformed_steps_are_rejected(raw):
    with pytest.raises(ValidationError):
        AgentStep.model_validate_json(raw)


@pytest.mark.parametrize("limit", [0, 26, -1])
def test_limit_bounds_are_enforced_by_the_grammar(limit):
    """Bounds live on StepArgs so the model cannot emit them, not so it is told off."""
    with pytest.raises(ValidationError):
        StepArgs(limit=limit)


def test_thought_is_capped():
    with pytest.raises(ValidationError):
        AgentStep(thought="x" * 201, tool=ToolName.ANSWER)
