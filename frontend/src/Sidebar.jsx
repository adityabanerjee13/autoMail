import React from "react";

const DAY = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function stamp(iso) {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? "" : DAY.format(parsed);
}

export default function Sidebar({ threads, activeId, onOpen, onNew, onDelete, busy }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <button className="btn wide" onClick={onNew} disabled={busy}>
          New chat
        </button>
      </div>
      <div className="thread-list">
        {threads.length === 0 && (
          <p className="muted" style={{ fontSize: 12, padding: "0 9px" }}>
            No conversations yet.
          </p>
        )}
        {threads.map((t) => (
          <div
            key={t.id}
            className={t.id === activeId ? "thread active" : "thread"}
            onClick={() => onOpen(t.id)}
          >
            <div className="thread-title">
              {t.title}
              <div className="thread-date">{stamp(t.updated_at)}</div>
            </div>
            <button
              className="thread-delete"
              title="Delete conversation"
              aria-label={`Delete ${t.title}`}
              onClick={(event) => {
                // Without this the click also selects the thread it is about
                // to delete, and the app briefly renders a 404.
                event.stopPropagation();
                onDelete(t.id);
              }}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
