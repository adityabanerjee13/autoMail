"""The agent turn: a bounded loop of constrained generations and tool calls.

The shape is deliberately dull. Ask the model for one JSON object, validate it,
run the tool it named, append the result, ask again. What makes it work on an
8B model is not the loop but the three things wrapped around it: the grammar
that makes an invalid tool name unspeakable, the error strings that tell the
model what to do rather than what went wrong, and the schema narrowing that
makes stopping mandatory rather than requested.

Every exit path writes a terminal row. A turn that ends without one leaves the
page polling forever, which is the failure mode this file works hardest to
avoid -- hence the ``finally`` and the restart sweep that backs it up.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from triage.agent import errors as err
from triage.agent.errors import STRIKE_CODES, ToolErrorCode, ToolFailure
from triage.agent.prompt import build_messages, estimate_messages, render_call
from triage.agent.schema import AgentStep, ToolName, agent_step_json_schema
from triage.agent.tools import REGISTRY, ToolResult, coerce_args
from triage.config import Settings, get_settings
from triage.db import chat
from triage.llm.client import LLMBadRequest, LLMClient, LLMUnavailable, _strip_fence

log = logging.getLogger("triage.agent.loop")

#: Tool calls before the model is forced to answer. Six covers the realistic
#: worst case (queue_status -> sync_mail -> run_queue -> queue_status -> answer)
#: and is already a ninety-second turn on this hardware; ten would be a worse
#: experience for no additional accuracy.
MAX_ITERATIONS = 6
MAX_ERROR_STRIKES = 3
TURN_TIMEOUT_S = 600
CONFIRM_TIMEOUT_S = 600

#: The agent's own output budget. Not client.MAX_OUTPUT_TOKENS, which is sized
#: for a Triage object of a few short fields; an `answer` carrying a table of
#: messages needs more room.
AGENT_MAX_OUTPUT_TOKENS = 768

TRUNCATION_MARKER = "<<truncated: finish_reason=length>>"

REPAIR_NUDGE = (
    "Your previous reply was not a valid object for the required schema. "
    "Reply with the JSON object only."
)
OUT_OF_ROOM = "You are out of room. Answer the user now with what you already know."
ITERATIONS_SPENT = (
    "You have used all {n} tool calls. Answer the user now with what you have found."
)
STRIKES_SPENT = (
    "Three of your tool calls were invalid. Stop and tell the user plainly what "
    "you could not do."
)


class TurnAborted(RuntimeError):
    """The turn cannot continue. Carries the sentence the user will read."""


async def run_turn(
    sessions: async_sessionmaker[AsyncSession],
    client: LLMClient,
    *,
    thread_id: int,
    turn_id: int,
    user_message: str,
    owner: str,
    confirm: Callable[[int], Awaitable[bool]],
    settings: Settings | None = None,
) -> None:
    """Run one turn to completion, writing every step as it happens.

    ``sessions`` is a sessionmaker, never a live session: this runs as a
    detached task that outlives the request that started it, and a request's
    session is closed the moment its response is sent.
    """
    settings = settings or get_settings()
    scratchpad: list[tuple[str, str]] = []
    strikes = 0
    force: list[ToolName] | None = None
    nudge: str | None = None
    started = time.perf_counter()

    try:
        for iteration in range(MAX_ITERATIONS + 1):
            if time.perf_counter() - started > TURN_TIMEOUT_S:
                raise TurnAborted(err.TURN_TIMEOUT)

            async with sessions() as session:
                history = await chat.history_for_model(
                    session, thread_id, before_turn=turn_id
                )

            # The last pass is not a tool call: it is the model being made to
            # answer. Narrowing the grammar is what makes that reliable -- an
            # instruction alone gets ignored by a model mid-plan.
            if iteration == MAX_ITERATIONS and force is None:
                force = [ToolName.ANSWER]
                nudge = ITERATIONS_SPENT.format(n=MAX_ITERATIONS)

            step, raw, latency = await _generate(
                client,
                owner=owner,
                history=history,
                user_message=user_message,
                scratchpad=scratchpad,
                force=force,
                nudge=nudge,
                settings=settings,
            )
            nudge = None

            if step.tool is ToolName.ANSWER:
                await _write_answer(sessions, thread_id, turn_id, step, raw, latency)
                return

            spec = REGISTRY[step.tool]
            args_dump = step.args.model_dump(exclude_none=True)

            # Validate before writing the row: a call that never happened
            # should not appear in the transcript as though it did.
            try:
                args = coerce_args(spec, step.args)
            except ValidationError as exc:
                strikes += 1
                scratchpad.append(
                    (render_call(step.tool.value, args_dump), err.bad_args(step.tool.value, exc))
                )
                await _write_tool_error(
                    sessions,
                    thread_id,
                    turn_id,
                    step,
                    ToolErrorCode.BAD_ARGS,
                    "corrected a bad tool call",
                )
                force, nudge = _maybe_force(strikes, force, nudge)
                continue

            row_id = await _write_tool_start(sessions, thread_id, turn_id, step, args_dump, spec)

            if spec.destructive:
                approved, expired = await _await_confirmation(confirm, row_id)
                if not approved:
                    async with sessions() as session:
                        await chat.finish_message(
                            session,
                            row_id,
                            status="declined",
                            content="expired" if expired else "cancelled",
                            error_code=ToolErrorCode.DECLINED.value,
                        )
                        await session.commit()
                    scratchpad.append(
                        (
                            render_call(step.tool.value, args_dump),
                            err.declined(step.tool.value, expired=expired),
                        )
                    )
                    # There is nothing left to explore after a "no" -- the next
                    # step should be telling the user it was cancelled.
                    force, nudge = [ToolName.ANSWER], None
                    continue

            code, result = await _run_tool(spec, args, sessions)
            if code is None:
                assert result is not None
                async with sessions() as session:
                    await chat.finish_message(
                        session,
                        row_id,
                        status="done",
                        content=result.summary,
                        tool_result={
                            "ok": True,
                            "text": result.text,
                            "summary": result.summary,
                            "meta": result.meta,
                        },
                    )
                    await session.commit()
                scratchpad.append(
                    (
                        render_call(step.tool.value, args_dump),
                        f"TOOL RESULT {step.tool.value}\n{result.text}",
                    )
                )
                continue

            detail = str(result)
            async with sessions() as session:
                await chat.finish_message(
                    session,
                    row_id,
                    status="error",
                    content=detail[:300],
                    error_code=code.value,
                    tool_result={"ok": False, "text": detail, "summary": code.value},
                )
                await session.commit()
            scratchpad.append((render_call(step.tool.value, args_dump), detail))
            if code in STRIKE_CODES:
                strikes += 1
                force, nudge = _maybe_force(strikes, force, nudge)

        # Unreachable: the iteration == MAX_ITERATIONS pass is forced to answer.
        raise TurnAborted(err.SCHEMA_FAILED)

    except TurnAborted as exc:
        await _write_failure(sessions, thread_id, turn_id, str(exc))
    except asyncio.CancelledError:
        await _write_failure(sessions, thread_id, turn_id, "stopped at your request")
        raise
    except Exception as exc:  # noqa: BLE001 - a turn must always terminate
        log.exception("agent turn failed")
        await _write_failure(sessions, thread_id, turn_id, f"Something went wrong: {exc!s}"[:300])


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


async def _generate(
    client: LLMClient,
    *,
    owner: str,
    history: list[list[dict[str, str]]],
    user_message: str,
    scratchpad: list[tuple[str, str]],
    force: list[ToolName] | None,
    nudge: str | None,
    settings: Settings,
) -> tuple[AgentStep, str, int]:
    """One step, with a single repair attempt. Mirrors ``LLMClient.classify``."""
    schema = agent_step_json_schema(only=force)
    started = time.perf_counter()
    last_raw = ""
    shrink = False

    for attempt in (1, 2):
        messages, _ = build_messages(
            owner=owner,
            history=history[len(history) // 2 :] if shrink else history,
            user_message=user_message,
            scratchpad=scratchpad[len(scratchpad) // 2 :] if shrink else scratchpad,
            max_iterations=MAX_ITERATIONS,
            nudge=REPAIR_NUDGE if attempt == 2 and not shrink else nudge,
        )
        estimate = estimate_messages(messages)
        try:
            raw, prompt_tokens = await client.complete_json(
                messages,
                schema=schema,
                max_tokens=AGENT_MAX_OUTPUT_TOKENS,
                # The repair attempt is always greedy, as in classify().
                temperature=0.0 if attempt == 2 else settings.llm_temperature,
            )
        except LLMBadRequest as exc:
            # The estimate was wrong, not the server. Halve the context and try
            # once; log loudly, because this is the signal that CHARS_PER_TOKEN
            # needs lowering.
            if shrink:
                raise TurnAborted(err.CONTEXT_FULL) from exc
            log.warning(
                "vLLM rejected a %d-token estimate (%s); shrinking and retrying",
                estimate,
                exc,
            )
            shrink = True
            continue
        except LLMUnavailable as exc:
            raise TurnAborted(err.unavailable(settings.vllm_base_url)) from exc

        # Worth watching for the first twenty steps of a live run: the gap
        # between these two numbers is what CHARS_PER_TOKEN is tuned against.
        log.debug("agent prompt estimate=%d actual=%s", estimate, prompt_tokens)
        last_raw = raw

        if TRUNCATION_MARKER in raw:
            # A cut-off object is a schema failure, not a transport one.
            continue
        try:
            step = AgentStep.model_validate_json(_strip_fence(raw))
        except (ValidationError, ValueError):
            continue
        return step, raw, int((time.perf_counter() - started) * 1000)

    log.warning("agent step failed schema validation twice: %.200s", last_raw)
    raise TurnAborted(err.SCHEMA_FAILED)


def _maybe_force(
    strikes: int, force: list[ToolName] | None, nudge: str | None
) -> tuple[list[ToolName] | None, str | None]:
    if strikes >= MAX_ERROR_STRIKES and force is None:
        return [ToolName.ANSWER], STRIKES_SPENT
    return force, nudge


# ---------------------------------------------------------------------------
# tool execution
# ---------------------------------------------------------------------------


async def _run_tool(spec, args, sessions) -> tuple[ToolErrorCode | None, ToolResult | str]:
    """Run one tool. Returns ``(None, result)`` or ``(code, model-facing text)``.

    No tool is ever retried here. Reads are idempotent but cost a full ~6k
    prefill to repeat; writes are not idempotent at all -- two enqueues are two
    jobs, which is the exact queue-inflation bug ``pending_job_for`` exists to
    prevent. The model may choose to call something again with different
    arguments; that is a decision, and it costs an iteration.
    """
    try:
        async with sessions() as session:
            return None, await asyncio.wait_for(spec.run(session, args), spec.timeout_s)
    except TimeoutError:
        return ToolErrorCode.TIMEOUT, err.tool_timeout(spec.name.value, spec.timeout_s)
    except ToolFailure as exc:
        return exc.code, err.format_tool_error(exc.code, exc.message)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a sentence
        log.exception("tool %s failed", spec.name.value)
        # str, never repr: the model has never seen a Python class name and
        # putting one in front of it invites imitation.
        detail = f"{exc!s}"[:200] or "no detail."
        return ToolErrorCode.FAILED, err.tool_failed(spec.name.value, detail)


async def _await_confirmation(
    confirm: Callable[[int], Awaitable[bool]], row_id: int
) -> tuple[bool, bool]:
    try:
        return await asyncio.wait_for(confirm(row_id), CONFIRM_TIMEOUT_S), False
    except TimeoutError:
        return False, True


# ---------------------------------------------------------------------------
# row writing
# ---------------------------------------------------------------------------


async def _write_tool_start(sessions, thread_id, turn_id, step, args_dump, spec) -> int:
    """Commit the tool row before running it.

    This is what makes the two-second poll show progress mid-turn instead of a
    spinner: each step becomes visible as it starts, and ``awaiting_confirm``
    needs a durable row anyway for the confirm POST to target.
    """
    async with sessions() as session:
        row_id = await chat.add_message(
            session,
            thread_id,
            role="tool",
            turn_id=turn_id,
            thought=step.thought,
            tool_name=step.tool.value,
            tool_args=args_dump,
            status="awaiting_confirm" if spec.destructive else "running",
            raw_step=step.model_dump_json(),
        )
        await session.commit()
    return row_id


async def _write_tool_error(sessions, thread_id, turn_id, step, code, summary) -> None:
    async with sessions() as session:
        await chat.add_message(
            session,
            thread_id,
            role="tool",
            turn_id=turn_id,
            thought=step.thought,
            tool_name=step.tool.value,
            content=summary,
            status="error",
            error_code=code.value,
            raw_step=step.model_dump_json(),
        )
        await session.commit()


async def _write_answer(sessions, thread_id, turn_id, step, raw, latency) -> None:
    text = (step.args.text or "").strip() or (
        "I could not put an answer together. Try asking again."
    )
    async with sessions() as session:
        await chat.add_message(
            session,
            thread_id,
            role="assistant",
            turn_id=turn_id,
            content=text,
            thought=step.thought,
            raw_step=raw,
            latency_ms=latency,
        )
        await chat.touch_thread(session, thread_id)
        await session.commit()


async def _write_failure(sessions, thread_id, turn_id, message) -> None:
    """Terminal row for every abort path.

    Written with ``status='error'`` so ``history_for_model`` skips it: a failed
    turn is not something to teach the model to imitate, but the user still
    needs to see what happened.
    """
    try:
        async with sessions() as session:
            await chat.add_message(
                session,
                thread_id,
                role="assistant",
                turn_id=turn_id,
                content=message,
                status="error",
                error_code=ToolErrorCode.FAILED.value,
            )
            await chat.touch_thread(session, thread_id)
            await session.commit()
    except Exception:  # noqa: BLE001 - nothing left to do if even this fails
        log.exception("could not write the failure row for thread %s", thread_id)


__all__ = ["MAX_ERROR_STRIKES", "MAX_ITERATIONS", "AGENT_MAX_OUTPUT_TOKENS", "run_turn"]
