"""Test doubles for the agent.

The repo had no mocks before this: parsing is tested against real .eml files and
the queue against a real Postgres, on the grounds that neither has a meaningful
fake. The agent does. Its state machine -- iteration caps, error strikes, schema
narrowing, confirmation -- is pure control flow that would otherwise only be
exercised by hand on a box where every LLM call costs fifteen seconds.

``ScriptedLLM`` records the schema it was handed on every call. That recording
is the point: it is the only way to assert that the loop narrowed the grammar
to ``answer`` when it was supposed to, which is the mechanism the whole
stopping story depends on.
"""

from __future__ import annotations

import json
from typing import Any

from triage.agent.schema import ToolName


class ScriptedLLM:
    """Stands in for LLMClient. Pops canned raw responses off a list."""

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        #: One entry per call: the schema passed in.
        self.schemas: list[dict[str, Any]] = []
        self.calls: list[list[dict[str, str]]] = []
        self.settings = _FakeSettings()

    async def complete_json(
        self, messages, *, schema, max_tokens, temperature=0.0
    ) -> tuple[str, int | None]:
        self.schemas.append(schema)
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies; the loop called once too often")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, 100

    # -- assertions helpers ------------------------------------------------

    def tools_offered(self, index: int) -> list[str]:
        """The tool enum the grammar allowed on call ``index``."""
        return self.schemas[index]["$defs"]["ToolName"]["enum"]


class _FakeSettings:
    vllm_base_url = "http://127.0.0.1:8000/v1"
    llm_temperature = 0.0


def step(tool: ToolName | str, thought: str = "thinking", **args: Any) -> str:
    """A raw model reply, as the loop will receive it."""
    name = tool.value if isinstance(tool, ToolName) else tool
    return json.dumps({"thought": thought, "tool": name, "args": args})


def answer(text: str = "here you go") -> str:
    return step(ToolName.ANSWER, text=text)


class FakeAgentRunner:
    """Records what the routes asked for, without spawning anything."""

    def __init__(self, *, busy: bool = False) -> None:
        self._busy = busy
        self.started: list[int] = []
        self.resolved: list[tuple[int, bool]] = []
        self.cancelled: list[int] = []
        self.resolve_result = True
        self.last_error: dict[int, str] = {}

    def is_running(self, thread_id: int) -> bool:
        return self._busy

    def is_busy(self) -> bool:
        return self._busy

    def start(self, thread_id: int, coro) -> bool:
        coro.close()
        if self._busy:
            return False
        self.started.append(thread_id)
        return True

    def resolve_confirmation(self, message_id: int, approved: bool) -> bool:
        self.resolved.append((message_id, approved))
        return self.resolve_result

    def cancel(self, thread_id: int) -> bool:
        self.cancelled.append(thread_id)
        return True

    async def wait_for_confirmation(self, message_id: int) -> bool:  # pragma: no cover
        return True


class FakeChatStore:
    """A dict-backed stand-in for db/chat.py, good enough for the loop.

    The loop only needs four things from the store: append a row, finish a row,
    read prior turns, and touch the thread. Everything else the real module does
    is for the templates.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._next_id = 1
        self.history: list[list[dict[str, str]]] = []

    async def add_message(self, session, thread_id, **kw) -> int:
        # Mirror the real signature's defaults, or a row written without an
        # explicit status comes back missing the key entirely.
        row = {
            "id": self._next_id,
            "thread_id": thread_id,
            "seq": len(self.rows) + 1,
            "status": "done",
            "content": "",
            **kw,
        }
        self._next_id += 1
        self.rows.append(row)
        return row["id"]

    async def finish_message(self, session, message_id, **kw) -> None:
        for row in self.rows:
            if row["id"] == message_id:
                row.update(kw)
                return

    async def history_for_model(self, session, thread_id, *, before_turn=None):
        return self.history

    async def touch_thread(self, session, thread_id, *, title=None) -> None:
        return None

    # -- assertion helpers -------------------------------------------------

    def roles(self) -> list[str]:
        return [r["role"] for r in self.rows]

    def by_role(self, role: str) -> list[dict]:
        return [r for r in self.rows if r["role"] == role]

    def final(self) -> dict | None:
        """The terminal assistant row, which every turn must write."""
        assistants = self.by_role("assistant")
        return assistants[-1] if assistants else None


class FakeSession:
    """Async context manager shaped like an AsyncSession. Records commits."""

    def __init__(self) -> None:
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:  # pragma: no cover
        return None

    async def close(self) -> None:  # pragma: no cover
        return None


def fake_sessionmaker() -> tuple[Any, FakeSession]:
    """A callable returning one shared FakeSession, plus that session."""
    session = FakeSession()

    def maker():
        return session

    return maker, session
