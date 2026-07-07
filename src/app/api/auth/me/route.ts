import { getCurrentUser } from "@/lib/auth";
import { handleApiError, json } from "@/lib/api-utils";

export async function GET() {
  try {
    const user = await getCurrentUser();
    return json({ user });
  } catch (error) {
    return handleApiError(error);
  }
}
