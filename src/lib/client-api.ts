"use client";

/** Thin client-side fetch wrapper. Throws Error with a user-facing message. */
export async function api<T>(
  path: string,
  init?: RequestInit & { json?: unknown }
): Promise<T> {
  const { json: jsonBody, ...rest } = init ?? {};
  const res = await fetch(path, {
    ...rest,
    headers: {
      ...(jsonBody !== undefined ? { "Content-Type": "application/json" } : {}),
      ...rest.headers,
    },
    ...(jsonBody !== undefined ? { body: JSON.stringify(jsonBody) } : {}),
  });

  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    // Non-JSON response (e.g. proxy error page)
  }
  if (!res.ok) {
    const message =
      data && typeof data === "object" && "error" in data && typeof data.error === "string"
        ? data.error
        : `Request failed (${res.status})`;
    throw new Error(message);
  }
  return data as T;
}

export function newIdempotencyKey(): string {
  return (
    globalThis.crypto?.randomUUID?.().replace(/-/g, "") ??
    `${Date.now()}${Math.random().toString(36).slice(2)}`
  );
}
