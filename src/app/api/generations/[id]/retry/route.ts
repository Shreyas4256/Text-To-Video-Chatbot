import { NextRequest } from "next/server";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { handleApiError, json, NotFoundError, RateLimitedError } from "@/lib/api-utils";
import { LIMITS, rateLimit } from "@/lib/rate-limit";
import { createGeneration } from "@/lib/generation-service";
import { toGenerationDto } from "@/lib/dto";
import { DEFAULT_SETTINGS, type AspectRatio, type Quality } from "@/lib/constants";

/** Create a fresh generation reusing a previous generation's prompt/settings. */
export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const user = await requireUser();
    const { id } = await params;

    const rl = rateLimit(
      `gen:${user.id}`,
      LIMITS.generation.limit,
      LIMITS.generation.windowMs
    );
    if (!rl.ok) throw new RateLimitedError(rl.retryAfterSec);

    const source = await db.videoGeneration.findFirst({
      where: { id, userId: user.id },
      include: { settings: true },
    });
    if (!source) throw new NotFoundError("Generation not found.");

    const generation = await createGeneration(user.id, {
      prompt: source.prompt,
      settings: source.settings
        ? {
            aspectRatio: source.settings.aspectRatio as AspectRatio,
            durationSec: source.settings.durationSec,
            style: source.settings.style,
            cameraMovement: source.settings.cameraMovement,
            motionStrength: source.settings.motionStrength,
            quality: source.settings.quality as Quality,
            seed: source.settings.seed ?? undefined,
            negativePrompt: source.negativePrompt ?? undefined,
          }
        : { ...DEFAULT_SETTINGS },
      conversationId: source.conversationId ?? undefined,
      enhance: !source.enhancedPrompt,
      precomputedEnhancedPrompt: source.enhancedPrompt ?? undefined,
    });

    if (generation.conversationId) {
      await db.message.create({
        data: {
          conversationId: generation.conversationId,
          role: "ASSISTANT",
          content: "Retrying that generation — new attempt submitted.",
          generationId: generation.id,
        },
      });
    }

    return json({ generation: toGenerationDto(generation) }, { status: 201 });
  } catch (error) {
    return handleApiError(error);
  }
}
