import { NextRequest } from "next/server";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { handleApiError, json, NotFoundError } from "@/lib/api-utils";
import { toConversationDto, toGenerationDto, toMessageDto } from "@/lib/dto";
import { generationInclude } from "@/lib/generation-service";

type Params = { params: Promise<{ id: string }> };

export async function GET(_req: NextRequest, { params }: Params) {
  try {
    const user = await requireUser();
    const { id } = await params;
    const conversation = await db.conversation.findFirst({
      where: { id, userId: user.id },
      include: {
        messages: { orderBy: { createdAt: "asc" }, take: 500 },
        generations: {
          include: generationInclude,
          orderBy: { createdAt: "asc" },
          take: 100,
        },
      },
    });
    if (!conversation) throw new NotFoundError("Conversation not found.");
    return json({
      conversation: toConversationDto(conversation),
      messages: conversation.messages.map(toMessageDto),
      generations: conversation.generations.map(toGenerationDto),
    });
  } catch (error) {
    return handleApiError(error);
  }
}

export async function DELETE(_req: NextRequest, { params }: Params) {
  try {
    const user = await requireUser();
    const { id } = await params;
    const result = await db.conversation.deleteMany({ where: { id, userId: user.id } });
    if (result.count === 0) throw new NotFoundError("Conversation not found.");
    return json({ ok: true });
  } catch (error) {
    return handleApiError(error);
  }
}
