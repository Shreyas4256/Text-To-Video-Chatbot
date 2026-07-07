import { NextRequest } from "next/server";
import { db } from "@/lib/db";
import { createSession } from "@/lib/auth";
import { verifyPassword } from "@/lib/password";
import { loginSchema } from "@/lib/validation";
import { ApiError, handleApiError, json, RateLimitedError } from "@/lib/api-utils";
import { LIMITS, rateLimit } from "@/lib/rate-limit";

export async function POST(req: NextRequest) {
  try {
    const body = loginSchema.parse(await req.json());

    const rl = rateLimit(`auth:login:${body.email}`, LIMITS.auth.limit, LIMITS.auth.windowMs);
    if (!rl.ok) throw new RateLimitedError(rl.retryAfterSec);

    const user = await db.user.findUnique({ where: { email: body.email } });
    // Same error for unknown email and wrong password — no account enumeration.
    if (!user || !(await verifyPassword(body.password, user.passwordHash))) {
      throw new ApiError("Incorrect email or password.", 401);
    }

    await createSession(user.id);
    return json({
      user: { id: user.id, email: user.email, name: user.name, createdAt: user.createdAt },
    });
  } catch (error) {
    return handleApiError(error);
  }
}
