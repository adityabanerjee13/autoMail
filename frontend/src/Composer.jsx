import React, { useEffect, useRef, useState } from "react";

const MAX_ROWS_PX = 180;

/**
 * The composer stays mounted and outside the transcript, so a half-typed
 * follow-up survives every poll while a turn is running.
 *
 * It also stays *enabled* while busy. A disabled textarea would swallow
 * whatever the user typed next, and the server already refuses a second
 * concurrent turn and says so -- which is more honest than a control that
 * silently does nothing.
 */
export default function Composer({ value, onChange, onSend, busy, onStop, note }) {
  const box = useRef(null);
  const [rows] = useState(2);

  // Grow with the text, up to a cap, then scroll internally.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_ROWS_PX)}px`;
  }, [value]);

  const submit = (event) => {
    event.preventDefault();
    if (value.trim()) onSend();
  };

  const onKeyDown = (event) => {
    // Enter sends, Shift+Enter makes a newline -- the convention every chat
    // client uses, and the reason this is not a plain form submit.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (value.trim()) onSend();
    }
  };

  return (
    <div className="composer">
      <form className="composer-inner" onSubmit={submit}>
        <textarea
          ref={box}
          rows={rows}
          value={value}
          placeholder="Ask about your mail, or tell it what to do…"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
        />
        {busy ? (
          <button type="button" className="btn" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" className="btn primary" disabled={!value.trim()}>
            Send
          </button>
        )}
      </form>
      {note && <div className="hint">{note}</div>}
    </div>
  );
}
