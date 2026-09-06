"""vLLM client. Guided JSON only -- no free-text parsing anywhere.

The prompt for a given version lives in exactly one file. ``classify_v1.jinja``
carries both the system and user halves, split on the ---SYSTEM--- /
---USER--- markers, so that ``prompt_version='v1'`` in a stored row maps to one
openable file.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from triage.config import Settings, get_settings
from triage.llm.schemas import Triage, example_payload, triage_json_schema
from triage.taxonomy import render_taxonomy

#: The object is a handful of short fields. Anything approaching this cap means
#: the model is rambling inside `reason`, not that the budget is too small.
MAX_OUTPUT_TOKENS = 512

_SPLIT = re.compile(r"^---(SYSTEM|USER)---\s*$", re.MULTILINE)


class LLMSchemaError(RuntimeError):
    """The model returned something that is not a valid Triage object.

    Carries the raw text so the dead-letter path can persist what actually came
    back -- a truncated generation is not reconstructable from anything else.
    """

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class LLMUnavailable(RuntimeError):
    """vLLM is not reachable. Retryable at the job level, not a bad message."""


class LLMBadRequest(LLMUnavailable):
    """vLLM rejected the request itself -- almost always a prompt over max-model-len.

    A subclass rather than a sibling on purpose: every existing ``except
    LLMUnavailable`` keeps catching this, so the classification path behaves
    exactly as it did. What the subclass buys is the agent loop, which must
    tell "the conversation grew past 8192 tokens" (shrink and retry) apart from
    "the server is down" (abort and say so). Before this existed both arrived
    as the same exception with a different string.
    """


@dataclass(slots=True)
class LLMResult:
    triage: Triage
    raw_response: str
    input_tokens: int | None
    latency_ms: int
    attempts: int


@lru_cache
def _env(prompt_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(prompt_dir),
        undefined=StrictUndefined,  # a missing context key is a bug, not a blank
        trim_blocks=False,
        keep_trailing_newline=True,
    )


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = AsyncOpenAI(
            base_url=self.settings.vllm_base_url,
            api_key="EMPTY",  # vLLM ignores it; the SDK requires one
            timeout=self.settings.llm_timeout_s,
            max_retries=0,  # retries are handled here, with our own policy
        )
        self._schema = triage_json_schema()

    # -- prompt ------------------------------------------------------------

    def render(self, context: dict[str, Any], version: str | None = None) -> tuple[str, str]:
        version = version or self.settings.prompt_version
        template = _env(str(self.settings.prompt_dir)).get_template(f"classify_{version}.jinja")
        rendered = template.render(
            taxonomy=render_taxonomy(),
            example_json=json.dumps(example_payload(), indent=2),
            **context,
        )
        return _split_prompt(rendered)

    # -- inference ---------------------------------------------------------

    async def classify(
        self, context: dict[str, Any], *, version: str | None = None
    ) -> LLMResult:
        """One guided-JSON call, with a single repair attempt at temperature 0.

        Guided decoding makes a schema failure rare -- when it happens it is
        almost always a truncated generation. One retry, then the job
        dead-letters and a human looks at raw_response.
        """
        system, user = self.render(context, version)
        started = time.perf_counter()
        last_raw = ""

        for attempt in (1, 2):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            if attempt == 2:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not a valid object for the required "
                            "schema. Reply with the JSON object only."
                        ),
                    }
                )

            raw, input_tokens = await self._call(
                messages,
                # The repair attempt is always greedy, whatever the configured
                # temperature is.
                temperature=self.settings.llm_temperature if attempt == 1 else 0.0,
            )
            last_raw = raw
            try:
                triage = Triage.model_validate_json(_strip_fence(raw))
            except (ValidationError, ValueError):
                continue
            return LLMResult(
                triage=triage,
                raw_response=raw,
                input_tokens=input_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
                attempts=attempt,
            )

        raise LLMSchemaError("model output failed Triage validation twice", last_raw)


    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> tuple[str, int | None]:
        """One structured-output call against an arbitrary schema.

        The agent loop's way in. It exists so that the agent inherits the
        tenacity policy, the truncation marker and the exception normalisation
        in ``_call`` rather than reimplementing all three slightly differently.
        Returns the raw text and the prompt token count; validating it against
        a model is the caller's job, because the caller is the one that knows
        what to do when it fails.
        """
        return await self._call(
            messages, temperature=temperature, schema=schema, max_tokens=max_tokens
        )

    def _extra_body(self, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        """Structured-output constraint, plus the reasoning-model workaround.

        The handoff specifies `guided_json`, which was the spelling until vLLM
        removed it (along with guided_regex/choice/grammar and
        --guided-decoding-backend) in 0.12.0. Same feature, same schema,
        current name. Note the failure mode if this is ever pointed at an older
        server: vLLM ignores an unknown extra_body field rather than rejecting
        it, so generation would be unconstrained and the Triage validation is
        what catches it -- as a schema error, not as a silently wrong label.

        enable_thinking=False matters just as much. Qwen3 opens with a <think>
        block; the JSON grammar forbids it from the first token, so the model
        emits whitespace instead, right up to max_tokens. It comes back as an
        opening brace, a key, a colon, and then hundreds of newlines -- which
        reads like a broken schema and is not. Verified on Qwen3-1.7B: same
        prompt, same schema, degenerate with thinking on, valid object with it
        off. Set LLM_DISABLE_THINKING=false for a model that has no such block.

        ``schema`` defaults to the Triage schema fixed at construction, so
        ``classify`` emits exactly the request it always did. The agent passes
        its own.
        """
        body: dict[str, Any] = {"structured_outputs": {"json": schema or self._schema}}
        if self.settings.llm_disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    @retry(
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        schema: dict[str, Any] | None = None,
        max_tokens: int = MAX_OUTPUT_TOKENS,
    ) -> tuple[str, int | None]:
        try:
            response = await self._client.chat.completions.create(
                model=self.settings.vllm_model_id,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self._extra_body(schema),
            )
        except (APIConnectionError, APITimeoutError):
            raise
        except BadRequestError as exc:
            # vLLM answered, and said no. On this deployment that is nearly
            # always a prompt longer than --max-model-len. Distinguishable from
            # a dead server, which matters to the agent and not to classify().
            raise LLMBadRequest(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - normalised for the caller
            raise LLMUnavailable(str(exc)) from exc

        choice = response.choices[0]
        raw = choice.message.content or ""
        if choice.finish_reason == "length":
            # Surfaces as a schema error with the truncated text preserved.
            raw = raw + "\n<<truncated: finish_reason=length>>"
        usage = getattr(response, "usage", None)
        return raw, getattr(usage, "prompt_tokens", None)

    async def health(self) -> dict[str, Any]:
        try:
            models = await self._client.models.list()
            served = [m.id for m in models.data]
            return {
                "ok": True,
                "served_models": served,
                "configured_model": self.settings.vllm_model_id,
                "model_present": self.settings.vllm_model_id in served,
            }
        except Exception as exc:  # noqa: BLE001 - health must not raise
            return {"ok": False, "error": str(exc)}


def _split_prompt(rendered: str) -> tuple[str, str]:
    parts = _SPLIT.split(rendered)
    # ['', 'SYSTEM', <system text>, 'USER', <user text>]
    sections: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i]] = parts[i + 1].strip()
    if "SYSTEM" not in sections or "USER" not in sections:
        raise ValueError("prompt template must contain ---SYSTEM--- and ---USER--- markers")
    return sections["SYSTEM"], sections["USER"]


def _strip_fence(raw: str) -> str:
    """Guided decoding should never emit a fence, but a cheap guard is free."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()
