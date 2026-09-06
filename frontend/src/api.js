// Every call goes to the same origin; the API has no auth because it binds to
// loopback. Errors are surfaced as thrown Error objects with the server's
// `detail`, which is what the UI shows -- FastAPI's default shape.

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch {
      // A non-JSON error body (a proxy page, an empty 502) is not worth
      // reporting verbatim; the status line above is more useful.
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

export const listThreads = () => request("/api/chat/threads");
export const createThread = () => request("/api/chat/threads", { method: "POST" });
export const getThread = (id) => request(`/api/chat/threads/${id}`);
export const deleteThread = (id) =>
  request(`/api/chat/threads/${id}`, { method: "DELETE" });

export const sendMessage = (id, text) =>
  request(`/api/chat/threads/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });

export const confirmTool = (id, messageId, approved) =>
  request(`/api/chat/threads/${id}/confirm`, {
    method: "POST",
    body: JSON.stringify({ message_id: messageId, approved }),
  });

export const stopTurn = (id) =>
  request(`/api/chat/threads/${id}/stop`, { method: "POST" });

export const retryTurn = (id) =>
  request(`/api/chat/threads/${id}/retry`, { method: "POST" });
