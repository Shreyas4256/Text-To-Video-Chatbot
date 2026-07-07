import { NextRequest } from "next/server";
import type { Prisma } from "@prisma/client";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { createGenerationSchema, listGenerationsSchema } from "@/lib/validation";
import { handleApiError, json, RateLimitedError } from "@/lib/api-utils";
import { LIMITS, rateLimit } from "@/lib/rate-limit";
import { createGeneration, generationInclude } from "@/lib/generation-service";
import { toGenerationDto } from "@/lib/dto";
import type { AspectRatio, Quality } from "@/lib/constants";

export async function POST(req: NextRequest) {
  try {
    const user = await requireUser();

    const rl = rateLimit(
      `gen:${user.id}`,
      LIMITS.generation.limit,
      LIMITS.generation.windowMs
    );
    if (!rl.ok) throw new RateLimitedError(rl.retryAfterSec);

    const body = createGenerationSchema.parse(await req.json());

    const generation = await createGeneration(user.id, {
      prompt: body.prompt,
      settings: {
        ...body.settings,
        aspectRatio: body.settings.aspectRatio as AspectRatio,
        quality: body.settings.quality as Quality,
      },
      conversationId: body.conversationId,
      idempotencyKey: body.idempotencyKey,
      enhance: body.enhance,
    });

    // Anchor the generation in its conversation so the chat renders a live
    // status card. This message reflects a genuinely submitted job.
    if (generation.conversationId) {
      await db.message.create({
        data: {
          conversationId: generation.conversationId,
          role: "ASSISTANT",
          content:
            generation.status === "FAILED"
              ? "The generation request could not be submitted."
              : "Generation submitted — the video will appear here when it's ready.",
          generationId: generation.id,
        },
      });
    }

    return json({ generation: toGenerationDto(generation) }, { status: 201 });
  } catch (error) {
    return handleApiError(error);
  }
}

export async function GET(req: NextRequest) {
  try {
    const user = await requireUser();
    const query = listGenerationsSchema.parse(
      Object.fromEntries(new URL(req.url).searchParams)
    );

    const where: Prisma.VideoGenerationWhereInput = {
      userId: user.id,
      ...(query.status ? { status: query.status } : {}),
      ...(query.search
        ? {
            OR: [
              { prompt: { contains: query.search, mode: "insensitive" } },
              { enhancedPrompt: { contains: query.search, mode: "insensitive" } },
            ],
          }
        : {}),
    };

    const orderBy: Prisma.VideoGenerationOrderByWithRelationInput[] =
      query.sort === "oldest"
        ? [{ createdAt: "asc" }]
        : query.sort === "duration"
          ? [{ settings: { durationSec: "desc" } }, { createdAt: "desc" }]
          : query.sort === "status"
            ? [{ status: "asc" }, { createdAt: "desc" }]
            : [{ createdAt: "desc" }];

    const items = await db.videoGeneration.findMany({
      where,
      include: generationInclude,
      orderBy,
      take: query.limit + 1,
      ...(query.cursor ? { cursor: { id: query.cursor }, skip: 1 } : {}),
    });

    const hasMore = items.length > query.limit;
    const page = hasMore ? items.slice(0, query.limit) : items;
    return json({
      generations: page.map(toGenerationDto),
      nextCursor: hasMore ? page[page.length - 1].id : null,
    });
  } catch (error) {
    return handleApiError(error);
  }
}
