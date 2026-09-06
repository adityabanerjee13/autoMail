"""The chat JSON API, which the React client is the only consumer of.

TestClient works here without a running Postgres because ``get_session`` is
overridden -- which is also the only way it *can* work on a Windows dev box,
where psycopg refuses TestClient's proactor event loop.

These replaced a set of HTML-fragment tests when the chat moved to React. The
rest of the app is still server-rendered htmx and still has to work with
JavaScript off; that rule applies to the ops console, not here.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from tests.fakes import FakeAgentRunner, FakeSession

from triage.api import deps
from triage.api.app import app
from triage.api.routes import chat as chat_routes

THREAD_ID = 7


class Row(dict):
    """The store returns ORM rows; the serializer reads them as attributes."""

    __getattr__ = dict.get


def make_row(**kw):
    base = {
        "id": 1,
        "seq": 1,
        "role": "user",
        "content": "",
        "thought": None,
        "tool_name": None,
        "tool_args": None,
        "tool_result": None,
        "status": "done",
        "error_code": None,
        "created_at": None,
    }
    return Row({**base, **kw})


@pytest.fixture
def client(monkeypatch):
    runner = FakeAgentRunner()
    monkeypatch.setattr(chat_routes, "agent_runner", runner)

    now = dt.datetime(2026, 9, 6, 12, 0, tzinfo=dt.UTC)
    state = {
        "rows": [],
        "thread": {"id": THREAD_ID, "title": "what is queued?", "updated_at": now},
    }

    async def get_thread(session, thread_id, owner):
        return state["thread"]

    async def messages_for_render(session, thread_id):
        return state["rows"]

    async def list_threads(session, owner, limit=50):
        return [state["thread"]] if state["thread"] else []

    async def add_message(session, thread_id, **kw):
        row = make_row(id=99, seq=len(state["rows"]) + 1, **kw)
        state["rows"].append(row)
        return 99

    async def create_thread(session, owner):
        return THREAD_ID

    async def delete_thread(session, thread_id, owner):
        state["thread"] = None
        return True

    async def noop(*a, **k):
        return None

    for name, fn in {
        "get_thread": get_thread,
        "messages_for_render": messages_for_render,
        "list_threads": list_threads,
        "add_message": add_message,
        "create_thread": create_thread,
        "delete_thread": delete_thread,
        "touch_thread": noop,
    }.items():
        monkeypatch.setattr(chat_routes.chat, name, fn)
    monkeypatch.setattr(chat_routes.chat, "owner_key", lambda *a, **k: "me@example.com")

    async def fake_session():
        yield FakeSession()

    app.dependency_overrides[deps.get_session] = fake_session
    with TestClient(app) as c:
        c.fake_runner = runner
        c.state = state
        yield c
    app.dependency_overrides.clear()


# -- threads ----------------------------------------------------------------


def test_listing_threads_reports_the_owner(client):
    body = client.get("/api/chat/threads").json()
    assert body["owner"] == "me@example.com"
    assert body["threads"][0]["id"] == THREAD_ID
    # Serialised, not a datetime -- the client formats it in the local zone.
    assert isinstance(body["threads"][0]["updated_at"], str)


def test_creating_a_thread_returns_its_id(client):
    r = client.post("/api/chat/threads")
    assert r.status_code == 201
    assert r.json()["id"] == THREAD_ID


def test_a_missing_thread_is_a_404(client):
    client.state["thread"] = None
    assert client.get(f"/api/chat/threads/{THREAD_ID}").status_code == 404


def test_deleting_a_thread_cancels_any_turn_on_it(client):
    """Leaving the task running would write rows into a thread that is gone."""
    assert client.delete(f"/api/chat/threads/{THREAD_ID}").status_code == 200
    assert client.fake_runner.cancelled == [THREAD_ID]


# -- transcript -------------------------------------------------------------


def test_the_thread_payload_carries_the_live_state(client):
    body = client.get(f"/api/chat/threads/{THREAD_ID}").json()
    assert body["thread"]["id"] == THREAD_ID
    assert body["busy"] is False
    assert body["queue_running"] in (True, False)
    # Sent once so the client can render a banner stating the real consequence
    # without duplicating the wording.
    assert "kept" in body["confirm_prompts"]["clear_queue"]


def test_only_destructive_tools_have_confirm_prompts(client):
    prompts = client.get(f"/api/chat/threads/{THREAD_ID}").json()["confirm_prompts"]
    assert set(prompts) == {"clear_queue", "dequeue_message"}


def test_tool_rows_expose_the_model_facing_text_but_not_the_meta(client):
    client.state["rows"].append(
        make_row(
            id=5,
            role="tool",
            tool_name="queue_status",
            tool_result={"ok": True, "text": "pending 3", "summary": "s", "meta": {"x": 1}},
        )
    )
    row = client.get(f"/api/chat/threads/{THREAD_ID}").json()["messages"][0]
    assert row["tool_text"] == "pending 3"
    assert "meta" not in row


def test_busy_elsewhere_is_distinguished_from_busy_here(client, monkeypatch):
    """One turn runs process-wide, so a busy *other* thread needs its own note."""
    busy = FakeAgentRunner(busy=True)
    busy.is_running = lambda tid: False
    monkeypatch.setattr(chat_routes, "agent_runner", busy)
    body = client.get(f"/api/chat/threads/{THREAD_ID}").json()
    assert body["busy"] is False
    assert body["busy_elsewhere"] is True


# -- turns ------------------------------------------------------------------


def test_posting_a_message_returns_202_and_starts_a_turn(client):
    r = client.post(f"/api/chat/threads/{THREAD_ID}/messages", json={"text": "what is queued?"})
    assert r.status_code == 202
    assert r.json()["started"] is True
    assert client.fake_runner.started == [THREAD_ID]


def test_an_empty_message_is_refused(client):
    r = client.post(f"/api/chat/threads/{THREAD_ID}/messages", json={"text": "   "})
    assert r.status_code == 422
    assert client.fake_runner.started == []


def test_a_second_turn_while_busy_is_refused_with_a_reason(client, monkeypatch):
    """Two concurrent ~6k-token prompts is the contention case that hurts."""
    busy = FakeAgentRunner(busy=True)
    monkeypatch.setattr(chat_routes, "agent_runner", busy)

    body = client.post(
        f"/api/chat/threads/{THREAD_ID}/messages", json={"text": "and now?"}
    ).json()
    assert body["started"] is False
    assert "Still replying" in body["note"]
    assert busy.started == []


def test_confirm_and_cancel_reach_the_runner(client):
    for approved in (True, False):
        client.post(
            f"/api/chat/threads/{THREAD_ID}/confirm",
            json={"message_id": 42, "approved": approved},
        )
    assert client.fake_runner.resolved == [(42, True), (42, False)]


def test_a_stale_confirmation_reports_unresolved_rather_than_failing(client):
    """A double click, or a tab left open across a restart."""
    client.fake_runner.resolve_result = False
    r = client.post(
        f"/api/chat/threads/{THREAD_ID}/confirm", json={"message_id": 42, "approved": True}
    )
    assert r.status_code == 200
    assert r.json()["resolved"] is False


def test_confirm_requires_a_message_id(client):
    r = client.post(f"/api/chat/threads/{THREAD_ID}/confirm", json={"approved": True})
    assert r.status_code == 422


def test_stop_cancels_the_turn(client):
    assert client.post(f"/api/chat/threads/{THREAD_ID}/stop").json()["cancelled"] is True
    assert client.fake_runner.cancelled == [THREAD_ID]


# -- the shell --------------------------------------------------------------


def test_the_spa_shell_is_served_for_deep_links(client):
    """/chat and /chat/<id> both return the bundle; routing is client-side."""
    if not chat_routes.SPA_INDEX.exists():
        pytest.skip("frontend not built; run `npm run build` in frontend/")
    for path in ("/chat", f"/chat/{THREAD_ID}"):
        r = client.get(path)
        assert r.status_code == 200
        assert 'id="root"' in r.text
        # Hashed asset names change on rebuild, so a cached shell would point
        # at a bundle that no longer exists.
        assert r.headers["cache-control"] == "no-store"


def test_an_unbuilt_frontend_says_how_to_build_it(client, monkeypatch):
    monkeypatch.setattr(chat_routes, "SPA_INDEX", chat_routes.SPA_INDEX / "missing")
    r = client.get("/chat")
    assert r.status_code == 503
    assert "npm run build" in r.json()["detail"]
