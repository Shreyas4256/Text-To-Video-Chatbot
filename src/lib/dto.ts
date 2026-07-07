import type { AspectRatio, Quality } from "@/lib/constants";
import type { GenerationWithRelations } from "@/lib/generation-service";
import type { GenerationDto, MessageDto, ConversationDto } from "@/lib/types";
import type { Conversation, Message } from "@prisma/client";

export function toGenerationDto(g: GenerationWithRelations): GenerationDto {
  return {
    id: g.id,
    prompt: g.prompt,
    enhancedPrompt: g.enhancedPrompt,
    negativePrompt: g.negativePrompt,
    status: g.status,
    videoUrl: g.videoUrl,
    thumbnailUrl: g.thumbnailUrl,
    errorMessage: g.errorMessage,
    conversationId: g.conversationId,
    provider: g.providerRequest?.provider ?? null,
    settings: g.settings
      ? {
          aspectRatio: g.settings.aspectRatio as AspectRatio,
          durationSec: g.settings.durationSec,
          style: g.settings.style,
          cameraMovement: g.settings.cameraMovement,
          motionStrength: g.settings.motionStrength,
          quality: g.settings.quality as Quality,
          seed: g.settings.seed ?? undefined,
          negativePrompt: g.negativePrompt ?? undefined,
        }
      : null,
    createdAt: g.createdAt.toISOString(),
    updatedAt: g.updatedAt.toISOString(),
  };
}

export function toMessageDto(m: Message): MessageDto {
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    generationId: m.generationId,
    createdAt: m.createdAt.toISOString(),
  };
}

export function toConversationDto(c: Conversation): ConversationDto {
  return {
    id: c.id,
    title: c.title,
    createdAt: c.createdAt.toISOString(),
    updatedAt: c.updatedAt.toISOString(),
  };
}
