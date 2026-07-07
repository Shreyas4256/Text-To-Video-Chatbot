/**
 * Simple in-memory sliding-window rate limiter.
 *
 * Suitable for a single-instance deployment (VPS, Docker, Railway).
 * For multi-instance/serverless deployments swap this for a shared store
 * (e.g. Redis / Upstash) behind the same function signature.
 */
const buckets = new Map<string, number[]>();

const MAX_TRACKED_KEYS = 10_000;

export interface RateLimitResult {
  ok: boolean;
  retryAfterSec: number;
}

export function rateLimit(
  key: string,
  limit: number,
  windowMs: number
): RateLimitResult {
  const now = Date.now();
  const windowStart = now - windowMs;

  let hits = buckets.get(key)?.filter((t) => t > windowStart) ?? [];
  if (hits.length >= limit) {
    const retryAfterSec = Math.ceil((hits[0] + windowMs - now) / 1000);
    buckets.set(key, hits);
    return { ok: false, retryAfterSec: Math.max(retryAfterSec, 1) };
  }
  hits = [...hits, now];
  buckets.set(key, hits);

  // Opportunistic cleanup to bound memory.
  if (buckets.size > MAX_TRACKED_KEYS) {
    for (const [k, v] of buckets) {
      if (v.every((t) => t <= windowStart)) buckets.delete(k);
    }
  }
  return { ok: true, retryAfterSec: 0 };
}

export const LIMITS = {
  /** Video generations per user */
  generation: { limit: 5, windowMs: 60_000 },
  /** Chat messages per user */
  message: { limit: 20, windowMs: 60_000 },
  /** Auth attempts per identifier */
  auth: { limit: 10, windowMs: 15 * 60_000 },
} as const;
