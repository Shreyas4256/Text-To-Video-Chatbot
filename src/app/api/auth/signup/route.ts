import { NextRequest } from "next/server";
import { db } from "@/lib/db";
import { createSession } from "@/lib/auth";
import { hashPassword } from "@/lib/password";
import { signupSchema } from "@/lib/validation";
import { ApiError, handleApiError, json, RateLimitedError } from "@/lib/api-utils";
import { LIMITS, rateLimit } from "@/lib/rate-limit";

export async function POST(req: NextRequest) {
  try {
    const body = signupSchema.parse(await req.json());

    const rl = rateLimit(`auth:signup:${body.email}`, LIMITS.auth.limit, LIMITS.auth.windowMs);
    if (!rl.ok) throw new RateLimitedError(rl.retryAfterSec);

    const existing = await db.user.findUnique({ where: { email: body.email } });
    if (existing) {
      throw new ApiError("An account with this email already exists. Try logging in.", 409);
    }

    const user = await db.user.create({
      data: {
        email: body.email,
        name: body.name,
        passwordHash: await hashPassword(body.password),
      },
      select: { id: true, email: true, name: true, createdAt: true },
    });
    await createSession(user.id);
    return json({ user }, { status: 201 });
  } catch (error) {
    return handleApiError(error);
  }
}
