import { NextRequest } from "next/server";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { sendMessageSchema } from "@/lib/validation";
import { handleApiError, json, NotFoundError, RateLimitedError } from "@/lib/api-utils";
import { LIMITS, rateLimit } from "@/lib/rate-limit";
import { runAssistantTurn } from "@/lib/assistant";
import { toMessageDto } from "@/lib/dto";
import type { AspectRatio, Quality } from "@/lib/constants";

/**
 * A chat turn: stores the user's message, runs the assistant (prompt/settings
 * refinement — never fake generation), stores and returns the reply plus the
 * updated draft for the settings panel.
 */
export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const user = await requireUser();
    const { id } = await params;

    const rl = rateLimit(`msg:${user.id}`, LIMITS.message.limit, LIMITS.message.windowMs);
    if (!rl.ok) throw new RateLimitedError(rl.retryAfterSec);

    const body = sendMessageSchema.parse(await req.json());

    const conversation = await db.conversation.findFirst({
      where: { id, userId: user.id },
      select: { id: true, title: true },
    });
    if (!conversation) throw new NotFoundError("Conversation not found.");

    const userMessage = await db.message.create({
      data: { conversationId: id, role: "USER", content: body.content },
    });

    const draftIn = body.draft
      ? {
          prompt: body.draft.prompt,
          settings: {
            ...body.draft.settings,
            aspectRatio: body.draft.settings.aspectRatio as AspectRatio,
            quality: body.draft.settings.quality as Quality,
          },
        }
      : null;
    const result = await runAssistantTurn(body.content, draftIn);

    const assistantMessage = await db.message.create({
      data: { conversationId: id, role: "ASSISTANT", content: result.reply },
    });

    // First message names the conversation.
    const titleUpdate =
      conversation.title === "New video"
        ? { title: body.content.slice(0, 60) + (body.content.length > 60 ? "…" : "") }
        : {};
    await db.conversation.update({
      where: { id },
      data: { ...titleUpdate, updatedAt: new Date() },
    });

    return json({
      userMessage: toMessageDto(userMessage),
      assistantMessage: toMessageDto(assistantMessage),
      draft: result.draft,
    });
  } catch (error) {
    return handleApiError(error);
  }
}
