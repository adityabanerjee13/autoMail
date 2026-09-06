"""What the model is told when a tool call goes wrong.

Every string here follows the same three-part shape: **what happened, why, and
what to do next**. The third part is the one that matters. An 8B model handed
``ValidationError: 1 validation error for ReadMessageArgs`` will try the same
call again; the same model handed "call find_messages first to get an id, then
call read_message again" will do that instead. The error message is not a log
line, it is the next instruction.

So: no tracebacks, no exception class names, no Python identifiers the model
has not already seen in the tool catalogue.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import ValidationError


class ToolErrorCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    BAD_ARGS = "bad_args"
    NOT_FOUND = "not_found"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DECLINED = "declined"


class ToolFailure(RuntimeError):
    """Raised by an executor when the tool cannot do its job.

    Carries a code so the loop can decide whether it counts as a strike --
    ``NOT_FOUND`` deliberately does not, because discovering that message 9999
    does not exist is a legitimate finding rather than a mistake.
    """

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


#: NOT_FOUND is absent on purpose. See ToolFailure.
STRIKE_CODES = frozenset(
    {
        ToolErrorCode.UNKNOWN_TOOL,
        ToolErrorCode.BAD_ARGS,
        ToolErrorCode.FAILED,
        ToolErrorCode.TIMEOUT,
    }
)


def format_tool_error(code: ToolErrorCode, detail: str) -> str:
    """The line appended to the scratchpad in place of a result."""
    return f"TOOL ERROR {code.value}: {detail}"


def unknown_tool(name: str, known: list[str]) -> str:
    return format_tool_error(
        ToolErrorCode.UNKNOWN_TOOL,
        f'"{name}" is not a tool. Use one of: {", ".join(known)}. '
        "Pick the closest one and call it again.",
    )


def bad_args(tool: str, exc: ValidationError) -> str:
    """Turn a pydantic error into at most three plain sentences.

    ``ValidationError.errors()`` is a list of dicts full of ``loc`` tuples and
    ``type`` slugs. None of that helps the model; the field name and a plain
    description of what was wrong do. Three is the cap because a model that got
    four fields wrong needs to be told to start over, not given a checklist.
    """
    problems = []
    for err in exc.errors()[:3]:
        field = ".".join(str(x) for x in err.get("loc", ())) or "an argument"
        problems.append(f"{field}: {err.get('msg', 'is not valid')}")
    return format_tool_error(
        ToolErrorCode.BAD_ARGS,
        f"{tool} was called with arguments it cannot use. "
        + "; ".join(problems)
        + ". Fix the arguments and call it again.",
    )


def missing_required(tool: str, field: str, hint: str) -> str:
    return format_tool_error(
        ToolErrorCode.BAD_ARGS, f"{tool} needs {field}, and you did not send one. {hint}"
    )


def tool_failed(tool: str, detail: str) -> str:
    return format_tool_error(
        ToolErrorCode.FAILED,
        f"{tool} could not run - {detail} "
        "Trying it again will not help. Tell the user what happened.",
    )


def tool_timeout(tool: str, seconds: float) -> str:
    return format_tool_error(
        ToolErrorCode.TIMEOUT,
        f"{tool} did not finish within {seconds:.0f} seconds. It may still be "
        "running in the background. Call queue_status to see the current state, "
        "or tell the user to check the queue page.",
    )


def declined(tool: str, *, expired: bool = False) -> str:
    why = "the user did not answer within 10 minutes" if expired else "the user said no"
    return format_tool_error(
        ToolErrorCode.DECLINED,
        f"{why} to {tool}. Do not call it again. Tell them it was cancelled.",
    )


# -- user-facing copy -------------------------------------------------------
# These are read by a person, not the model, so they say what to do about it in
# the operator's terms. This box has no ops team; the user is the ops team.

SCHEMA_FAILED = (
    "The model did not produce a usable action. This usually means the local "
    "model is overloaded or the reply was cut off."
)
CONTEXT_FULL = "This conversation has grown too long for the model. Start a new chat."
TURN_TIMEOUT = "This reply took too long and was stopped."
INTERRUPTED = "interrupted by a restart"


def unavailable(base_url: str) -> str:
    return (
        f"The local model is not reachable at {base_url}. "
        "Check that vLLM is running."
    )
