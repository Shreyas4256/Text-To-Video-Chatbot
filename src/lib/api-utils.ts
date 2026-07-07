import { NextResponse } from "next/server";
import { ZodError } from "zod";
import { UnauthorizedError } from "@/lib/auth";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class NotFoundError extends ApiError {
  constructor(message = "Not found") {
    super(message, 404);
  }
}

export class RateLimitedError extends ApiError {
  constructor(public retryAfterSec: number) {
    super(
      `You're doing that too fast. Try again in ${retryAfterSec} second${retryAfterSec === 1 ? "" : "s"}.`,
      429
    );
  }
}

/**
 * Convert internal errors into safe user-facing JSON responses.
 * Internal details are logged server-side, never returned to the client.
 */
export function handleApiError(error: unknown): NextResponse {
  if (error instanceof ZodError) {
    const first = error.issues[0];
    return NextResponse.json(
      { error: first?.message ?? "Invalid input", field: first?.path.join(".") },
      { status: 400 }
    );
  }
  if (error instanceof UnauthorizedError) {
    return NextResponse.json({ error: error.message }, { status: 401 });
  }
  if (error instanceof RateLimitedError) {
    return NextResponse.json(
      { error: error.message },
      { status: 429, headers: { "Retry-After": String(error.retryAfterSec) } }
    );
  }
  if (error instanceof ApiError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  console.error("[api] Unhandled error:", error);
  return NextResponse.json(
    { error: "Something went wrong on our side. Please try again." },
    { status: 500 }
  );
}

export function json<T>(data: T, init?: ResponseInit): NextResponse {
  return NextResponse.json(data, init);
}
