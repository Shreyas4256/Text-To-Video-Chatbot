import { NextRequest } from "next/server";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { createConversationSchema } from "@/lib/validation";
import { handleApiError, json } from "@/lib/api-utils";
import { toConversationDto } from "@/lib/dto";

export async function GET() {
  try {
    const user = await requireUser();
    const conversations = await db.conversation.findMany({
      where: { userId: user.id },
      orderBy: { updatedAt: "desc" },
      take: 100,
    });
    return json({ conversations: conversations.map(toConversationDto) });
  } catch (error) {
    return handleApiError(error);
  }
}

export async function POST(req: NextRequest) {
  try {
    const user = await requireUser();
    const body = createConversationSchema.parse(await req.json().catch(() => ({})));
    const conversation = await db.conversation.create({
      data: { userId: user.id, title: body.title ?? "New video" },
    });
    return json({ conversation: toConversationDto(conversation) }, { status: 201 });
  } catch (error) {
    return handleApiError(error);
  }
}
