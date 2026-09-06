import React, { useEffect, useLayoutEffect, useRef } from "react";
import Message from "./Message.jsx";

/** Within this many pixels of the bottom counts as "following the conversation". */
const STICK_PX = 120;

const SUGGESTIONS = [
  "what's in my queue?",
  "show me important mail from the last 3 days",
  "sync my mail and classify anything new",
  "how many emails do I have, and how many are classified?",
];

/**
 * The scrolling half of the chat.
 *
 * Autoscroll is conditional on purpose. A turn writes rows for a minute or
 * more, and unconditionally jumping to the bottom on every poll would rip the
 * page away from someone who scrolled up to read what a tool returned. So the
 * scroll position is only pinned when the user was already near the bottom;
 * scroll up and it leaves you alone until you come back down.
 */
export default function Transcript({
  messages,
  busy,
  confirmPrompts,
  onConfirm,
  onRetry,
  onSuggest,
  pending,
}) {
  const scroller = useRef(null);
  const stick = useRef(true);

  // Recorded before paint, because reading scrollTop after React has already
  // inserted the new rows would measure the post-growth position and conclude
  // we had scrolled up.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    stick.current = distance < STICK_PX;
  });

  useEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_PX;
  };

  return (
    <div className="transcript" ref={scroller} onScroll={onScroll}>
      {messages.length === 0 && !busy ? (
        <div className="empty">
          <p>Ask about your mail, or tell the agent what to do.</p>
          <ul>
            {SUGGESTIONS.map((s) => (
              <li key={s}>
                <button
                  className="btn small"
                  style={{ margin: "3px 0" }}
                  onClick={() => onSuggest(s)}
                >
                  {s}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="turns">
          {messages.map((row) => (
            <Message
              key={row.id}
              row={row}
              confirmPrompts={confirmPrompts}
              onConfirm={onConfirm}
              onRetry={onRetry}
              pending={pending}
            />
          ))}
          {busy && (
            <div className="thinking">
              <span className="dot" />
              <span>thinking…</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
