import { destroySession } from "@/lib/auth";
import { handleApiError, json } from "@/lib/api-utils";

export async function POST() {
  try {
    await destroySession();
    return json({ ok: true });
  } catch (error) {
    return handleApiError(error);
  }
}
