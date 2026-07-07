import type { Prisma } from "@prisma/client";
import { db } from "@/lib/db";
import { enhancePrompt } from "@/lib/assistant";
import type { DraftSettings } from "@/lib/constants";
import { getEnv } from "@/lib/env";
import { getVideoProvider } from "@/lib/video-providers/provider-factory";
import { ProviderError } from "@/lib/video-providers/provider-interface";

/** Minimum interval between polls of the provider for the same job. */
const POLL_INTERVAL_MS = 3_000;

export const generationInclude = {
  settings: true,
  providerRequest: {
    select: { provider: true, providerJobId: true, lastPolledAt: true, pollCount: true },
  },
} satisfies Prisma.VideoGenerationInclude;

export type GenerationWithRelations = Prisma.VideoGenerationGetPayload<{
  include: typeof generationInclude;
}>;

export interface CreateGenerationInput {
  prompt: string;
  settings: DraftSettings;
  conversationId?: string;
  idempotencyKey?: string;
  enhance: boolean;
  /** Reuse an already-enhanced prompt (e.g. on retry) instead of re-enhancing. */
  precomputedEnhancedPrompt?: string;
}

/**
 * Create a generation record and submit the job to the configured provider.
 * Retry-safe: an idempotencyKey that already exists returns the original
 * generation instead of creating a duplicate provider job.
 */
export async function createGeneration(
  userId: string,
  input: CreateGenerationInput
): Promise<GenerationWithRelations> {
  if (input.idempotencyKey) {
    const existing = await db.videoGeneration.findUnique({
      where: { idempotencyKey: input.idempotencyKey },
      include: generationInclude,
    });
    if (existing) {
      if (existing.userId !== userId) {
        // Key collision across users — treat as a fresh request without the key.
        input = { ...input, idempotencyKey: undefined };
      } else {
        return existing;
      }
    }
  }

  if (input.conversationId) {
    const conversation = await db.conversation.findFirst({
      where: { id: input.conversationId, userId },
      select: { id: true },
    });
    if (!conversation) input = { ...input, conversationId: undefined };
  }

  const enhancedPrompt =
    input.precomputedEnhancedPrompt ??
    (input.enhance ? await enhancePrompt(input.prompt, input.settings) : null);

  const generation = await db.videoGeneration.create({
    data: {
      userId,
      conversationId: input.conversationId,
      prompt: input.prompt,
      enhancedPrompt,
      negativePrompt: input.settings.negativePrompt,
      idempotencyKey: input.idempotencyKey,
      status: "QUEUED",
      settings: {
        create: {
          aspectRatio: input.settings.aspectRatio,
          durationSec: input.settings.durationSec,
          style: input.settings.style,
          cameraMovement: input.settings.cameraMovement,
          motionStrength: input.settings.motionStrength,
          quality: input.settings.quality,
          seed: input.settings.seed,
        },
      },
    },
    include: generationInclude,
  });

  const provider = getVideoProvider();
  try {
    const env = getEnv();
    const webhookUrl =
      env.APP_URL && env.WEBHOOK_SECRET
        ? `${env.APP_URL}/api/webhooks/video/${provider.name}?token=${env.WEBHOOK_SECRET}&generationId=${generation.id}`
        : undefined;

    const job = await provider.createJob({
      prompt: enhancedPrompt ?? input.prompt,
      negativePrompt: input.settings.negativePrompt,
      aspectRatio: input.settings.aspectRatio,
      durationSec: input.settings.durationSec,
      motionStrength: input.settings.motionStrength,
      quality: input.settings.quality,
      seed: input.settings.seed,
      webhookUrl,
    });

    const [updated] = await db.$transaction([
      db.videoGeneration.update({
        where: { id: generation.id },
        data: {
          providerRequest: {
            create: {
              provider: provider.name,
              providerJobId: job.providerJobId,
              requestPayload: job.requestPayload as Prisma.InputJsonValue,
            },
          },
        },
        include: generationInclude,
      }),
      db.usageRecord.create({
        data: { userId, kind: "video_generation" },
      }),
    ]);
    return updated;
  } catch (error) {
    const message =
      error instanceof ProviderError
        ? "The video provider rejected this request. Please adjust your prompt or try again later."
        : "Could not submit the generation request. Please try again.";
    console.error(`[generation] Provider submit failed for ${generation.id}:`, error);
    return db.videoGeneration.update({
      where: { id: generation.id },
      data: { status: "FAILED", errorMessage: message },
      include: generationInclude,
    });
  }
}

/**
 * Poll-on-read status sync. If the generation is still pending and hasn't
 * been polled within POLL_INTERVAL_MS, ask the provider for fresh status
 * and persist any transition. Terminal states are returned as-is.
 */
export async function syncGenerationStatus(
  generation: GenerationWithRelations
): Promise<GenerationWithRelations> {
  if (generation.status === "COMPLETED" || generation.status === "FAILED") {
    return generation;
  }
  const request = generation.providerRequest;
  if (!request) {
    // Submitted record without a provider job — submission failed mid-flight.
    return db.videoGeneration.update({
      where: { id: generation.id },
      data: {
        status: "FAILED",
        errorMessage: "The generation was never submitted to the provider.",
      },
      include: generationInclude,
    });
  }
  if (
    request.lastPolledAt &&
    Date.now() - request.lastPolledAt.getTime() < POLL_INTERVAL_MS
  ) {
    return generation;
  }

  await db.providerRequest.update({
    where: { generationId: generation.id },
    data: { lastPolledAt: new Date(), pollCount: { increment: 1 } },
  });

  try {
    const provider = getVideoProvider();
    const status = await provider.getJobStatus(request.providerJobId);
    const statusMap = {
      queued: "QUEUED",
      processing: "PROCESSING",
      completed: "COMPLETED",
      failed: "FAILED",
    } as const;
    const nextStatus = statusMap[status.state];
    // generation.status is non-terminal here, so an equal status means no
    // transition happened — skip the write.
    if (nextStatus === generation.status) {
      return generation;
    }
    return db.videoGeneration.update({
      where: { id: generation.id },
      data: {
        status: nextStatus,
        videoUrl: status.videoUrl,
        thumbnailUrl: status.thumbnailUrl,
        errorMessage: status.errorMessage,
      },
      include: generationInclude,
    });
  } catch (error) {
    // Transient polling failures leave the generation pending; the next
    // poll will retry. Log for observability.
    console.error(`[generation] Poll failed for ${generation.id}:`, error);
    return generation;
  }
}
