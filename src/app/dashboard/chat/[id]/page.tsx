import { notFound, redirect } from "next/navigation";
import { db } from "@/lib/db";
import { getCurrentUser } from "@/lib/auth";
import { generationInclude } from "@/lib/generation-service";
import { toConversationDto, toGenerationDto, toMessageDto } from "@/lib/dto";
import { ChatView } from "@/components/chat/chat-view";

export default async function ChatPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const user = await getCurrentUser();
  if (!user) redirect("/login");
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
  if (!conversation) notFound();

  return (
    <ChatView
      conversation={toConversationDto(conversation)}
      initialMessages={conversation.messages.map(toMessageDto)}
      initialGenerations={conversation.generations.map(toGenerationDto)}
    />
  );
}
