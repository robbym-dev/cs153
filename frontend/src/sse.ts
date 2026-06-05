import type { ProgressEvent } from "./types";

/** POST a multipart form and consume an SSE response, dispatching each event. */
export async function postSSE(
  url: string,
  body: FormData,
  onEvent: (e: ProgressEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(url, { method: "POST", body, signal });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      if (j?.detail) detail = j.detail;
    } catch {
      // ignore
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line.
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const dataLines = raw
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).replace(/^ /, ""));
      if (dataLines.length === 0) continue;
      try {
        onEvent(JSON.parse(dataLines.join("\n")) as ProgressEvent);
      } catch (err) {
        console.warn("malformed SSE event", raw, err);
      }
    }
  }
}
