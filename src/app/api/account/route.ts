import { NextRequest } from "next/server";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { hashPassword, verifyPassword } from "@/lib/password";
import { updateAccountSchema } from "@/lib/validation";
import { ApiError, handleApiError, json } from "@/lib/api-utils";

export async function PATCH(req: NextRequest) {
  try {
    const user = await requireUser();
    const body = updateAccountSchema.parse(await req.json());

    const data: { name?: string; passwordHash?: string } = {};
    if (body.name !== undefined) data.name = body.name;

    if (body.newPassword) {
      const full = await db.user.findUniqueOrThrow({ where: { id: user.id } });
      if (!(await verifyPassword(body.currentPassword ?? "", full.passwordHash))) {
        throw new ApiError("Current password is incorrect.", 400);
      }
      data.passwordHash = await hashPassword(body.newPassword);
    }

    const updated = await db.user.update({
      where: { id: user.id },
      data,
      select: { id: true, email: true, name: true, createdAt: true },
    });

    // Changing the password invalidates every other session.
    if (data.passwordHash) {
      await db.session.deleteMany({ where: { userId: user.id } });
      const { createSession } = await import("@/lib/auth");
      await createSession(user.id);
    }
    return json({ user: updated });
  } catch (error) {
    return handleApiError(error);
  }
}
