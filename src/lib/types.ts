import type { DraftSettings, GenerationStatusValue } from "@/lib/constants";

/** Client-safe DTOs shared between API routes and UI components. */

export interface GenerationDto {
  id: string;
  prompt: string;
  enhancedPrompt: string | null;
  negativePrompt: string | null;
  status: GenerationStatusValue;
  videoUrl: string | null;
  thumbnailUrl: string | null;
  errorMessage: string | null;
  conversationId: string | null;
  provider: string | null;
  settings: DraftSettings | null;
  createdAt: string;
  updatedAt: string;
}

export interface MessageDto {
  id: string;
  role: "USER" | "ASSISTANT";
  content: string;
  generationId: string | null;
  createdAt: string;
}

export interface ConversationDto {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}

export interface DraftDto {
  prompt: string;
  settings: DraftSettings;
}

export interface UserDto {
  id: string;
  email: string;
  name: string | null;
  createdAt: string;
}
