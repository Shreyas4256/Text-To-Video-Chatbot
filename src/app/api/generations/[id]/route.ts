import { NextRequest } from "next/server";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { handleApiError, json, NotFoundError } from "@/lib/api-utils";
import { generationInclude, syncGenerationStatus } from "@/lib/generation-service";
import { toGenerationDto } from "@/lib/dto";

type Params = { params: Promise<{ id: string }> };

/** Fetch one generation; pending jobs are synced against the provider. */
export async function GET(_req: NextRequest, { params }: Params) {
  try {
    const user = await requireUser();
    const { id } = await params;
    const generation = await db.videoGeneration.findFirst({
      where: { id, userId: user.id },
      include: generationInclude,
    });
    if (!generation) throw new NotFoundError("Generation not found.");
    const synced = await syncGenerationStatus(generation);
    return json({ generation: toGenerationDto(synced) });
  } catch (error) {
    return handleApiError(error);
  }
}

export async function DELETE(_req: NextRequest, { params }: Params) {
  try {
    const user = await requireUser();
    const { id } = await params;
    const result = await db.videoGeneration.deleteMany({
      where: { id, userId: user.id },
    });
    if (result.count === 0) throw new NotFoundError("Generation not found.");
    return json({ ok: true });
  } catch (error) {
    return handleApiError(error);
  }
}
