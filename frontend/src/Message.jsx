import React from "react";

const TIME = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
});

function when(iso) {
  if (!iso) return "";
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? "" : TIME.format(parsed);
}

/**
 * One tool call: the model's caption, a chip naming the tool, and the result
 * it was given, collapsed.
 *
 * The caption above the chip is doing real work. A turn is up to two minutes on
 * this hardware, and "checking what is queued" is what makes that time read as
 * progress rather than as a hang.
 */
function ToolStep({ row, confirmPrompt, onConfirm, pending }) {
  const failed = row.status === "error";
  const awaiting = row.status === "awaiting_confirm";

  return (
    <div className="step">
      {row.thought && <div className="thought">{row.thought}</div>}

      <span className={failed ? "chip failed" : "chip"}>
        <span className="tool">{row.tool_name}</span>
        {row.content && (
          <>
            <span className="sep">·</span>
            <span>{row.content}</span>
          </>
        )}
        {row.status === "running" && (
          <>
            <span className="sep">·</span>
            <span className="muted">running…</span>
          </>
        )}
      </span>

      {awaiting && (
        <div className="confirm">
          <p>{confirmPrompt || "Run this?"}</p>
          <div className="actions">
            <button
              className="btn primary"
              disabled={pending}
              onClick={() => onConfirm(row.id, true)}
            >
              Yes, do it
            </button>
            <button
              className="btn"
              disabled={pending}
              onClick={() => onConfirm(row.id, false)}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {row.tool_text && (
        <details>
          <summary>what the model was told</summary>
          <pre>{row.tool_text}</pre>
        </details>
      )}
    </div>
  );
}

export default function Message({ row, confirmPrompts, onConfirm, onRetry, pending }) {
  if (row.role === "user") {
    return (
      <div className="row me">
        <div className="bubble me" title={when(row.created_at)}>
          {row.content}
        </div>
      </div>
    );
  }

  if (row.role === "assistant") {
    const failed = row.status === "error";
    return (
      <div className="row">
        <div className={failed ? "bubble failed" : "bubble agent"}>
          {row.content}
          {failed && (
            <div style={{ marginTop: 10 }}>
              <button className="btn small" onClick={onRetry} disabled={pending}>
                Try again
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <ToolStep
      row={row}
      confirmPrompt={confirmPrompts[row.tool_name]}
      onConfirm={onConfirm}
      pending={pending}
    />
  );
}
