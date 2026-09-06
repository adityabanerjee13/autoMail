import React, { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api.js";
import Composer from "./Composer.jsx";
import Sidebar from "./Sidebar.jsx";
import Transcript from "./Transcript.jsx";

/** Matches the server's own poll cadence; a turn writes a row every few seconds. */
const POLL_MS = 2000;

const NAV = [
  ["/", "messages"],
  ["/queue", "queue"],
  ["/review", "review"],
  ["/status", "status"],
];

/** /chat/12 -> 12. The shell is served for both /chat and /chat/<id>. */
function threadIdFromPath() {
  const match = window.location.pathname.match(/^\/chat\/(\d+)/);
  return match ? Number(match[1]) : null;
}

export default function App() {
  const [threads, setThreads] = useState([]);
  const [owner, setOwner] = useState("");
  const [activeId, setActiveId] = useState(threadIdFromPath);
  const [thread, setThread] = useState(null);
  const [draft, setDraft] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  const busy = Boolean(thread?.busy);

  const refreshThreads = useCallback(async () => {
    try {
      const data = await api.listThreads();
      setThreads(data.threads);
      setOwner(data.owner);
      return data.threads;
    } catch (e) {
      setError(e.message);
      return [];
    }
  }, []);

  const loadThread = useCallback(async (id) => {
    if (!id) {
      setThread(null);
      return;
    }
    try {
      setThread(await api.getThread(id));
      setError("");
    } catch (e) {
      if (e.status === 404) {
        // Deleted in another tab, or a stale deep link.
        setThread(null);
        setActiveId(null);
        window.history.replaceState({}, "", "/chat");
      } else {
        setError(e.message);
      }
    }
  }, []);

  // First load: pick the newest thread when the URL does not name one.
  useEffect(() => {
    (async () => {
      const list = await refreshThreads();
      const fromPath = threadIdFromPath();
      const id = fromPath ?? list[0]?.id ?? null;
      if (id) {
        setActiveId(id);
        window.history.replaceState({}, "", `/chat/${id}`);
      }
    })();
  }, [refreshThreads]);

  useEffect(() => {
    loadThread(activeId);
  }, [activeId, loadThread]);

  // Poll only while a turn is in flight, and stop as soon as it lands -- the
  // same conditional-polling shape the ops console uses. An idle chat makes no
  // requests at all.
  useEffect(() => {
    if (!activeId || !busy) return undefined;
    const timer = setInterval(() => loadThread(activeId), POLL_MS);
    return () => clearInterval(timer);
  }, [activeId, busy, loadThread]);

  // Refresh the sidebar on the busy -> idle edge. It cannot be done inside the
  // poll: the poll is torn down by the very state change that ends the turn,
  // so the last tick never observes it, and the thread's new timestamp (and
  // its ordering) would stay stale until a manual reload.
  const wasBusy = useRef(false);
  useEffect(() => {
    if (wasBusy.current && !busy) refreshThreads();
    wasBusy.current = busy;
  }, [busy, refreshThreads]);

  // Browser back/forward between threads.
  useEffect(() => {
    const onPop = () => setActiveId(threadIdFromPath());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const open = (id) => {
    if (id === activeId) return;
    setActiveId(id);
    setNote("");
    window.history.pushState({}, "", `/chat/${id}`);
  };

  const newChat = async () => {
    setPending(true);
    try {
      const { id } = await api.createThread();
      await refreshThreads();
      setActiveId(id);
      setNote("");
      window.history.pushState({}, "", `/chat/${id}`);
    } catch (e) {
      setError(e.message);
    } finally {
      setPending(false);
    }
  };

  const send = async (text) => {
    const body = (text ?? draft).trim();
    if (!body) return;

    let id = activeId;
    setPending(true);
    try {
      if (!id) {
        id = (await api.createThread()).id;
        setActiveId(id);
        window.history.replaceState({}, "", `/chat/${id}`);
      }
      const result = await api.sendMessage(id, body);
      if (result.started) {
        setDraft("");
        setNote("");
      } else {
        // Refused because something else is running. The draft is kept so
        // nothing the user typed is lost.
        setNote(result.note || "");
      }
      await loadThread(id);
      await refreshThreads();
    } catch (e) {
      setError(e.message);
    } finally {
      setPending(false);
    }
  };

  const act = async (fn) => {
    setPending(true);
    try {
      await fn();
      await loadThread(activeId);
    } catch (e) {
      setError(e.message);
    } finally {
      setPending(false);
    }
  };

  const confirm = (messageId, approved) =>
    act(async () => {
      const { resolved } = await api.confirmTool(activeId, messageId, approved);
      if (!resolved) setNote("That confirmation has already been answered.");
    });

  const remove = async (id) => {
    if (!window.confirm("Delete this conversation?")) return;
    await act(async () => {
      await api.deleteThread(id);
      const list = await refreshThreads();
      if (id === activeId) {
        const next = list[0]?.id ?? null;
        setActiveId(next);
        window.history.replaceState({}, "", next ? `/chat/${next}` : "/chat");
      }
    });
  };

  const messages = thread?.messages ?? [];
  const hint =
    note ||
    (thread?.busy_elsewhere && "Another conversation is being answered.") ||
    (thread?.queue_running &&
      "The queue is classifying right now — replies will be slower.") ||
    "";

  return (
    <div className="app">
      <header className="topbar">
        <h1>triage</h1>
        {NAV.map(([href, label]) => (
          <a key={href} href={href}>
            {label}
          </a>
        ))}
        <a href="/chat" className="active">
          chat
        </a>
        <span className="spacer" />
        {owner && <span className="owner">{owner}</span>}
      </header>

      <div className="body">
        <Sidebar
          threads={threads}
          activeId={activeId}
          onOpen={open}
          onNew={newChat}
          onDelete={remove}
          busy={pending}
        />

        <main className="chat">
          {error && (
            <div style={{ padding: "14px 20px 0" }}>
              <div className="banner">{error}</div>
            </div>
          )}

          <Transcript
            messages={messages}
            busy={busy}
            confirmPrompts={thread?.confirm_prompts ?? {}}
            onConfirm={confirm}
            onRetry={() => act(() => api.retryTurn(activeId))}
            onSuggest={(text) => send(text)}
            pending={pending}
          />

          <Composer
            value={draft}
            onChange={setDraft}
            onSend={() => send()}
            busy={busy}
            onStop={() => act(() => api.stopTurn(activeId))}
            note={hint}
          />
        </main>
      </div>
    </div>
  );
}
