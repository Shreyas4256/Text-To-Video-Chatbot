import { z } from "zod";
import { ASPECT_RATIOS, DURATIONS, QUALITIES, STYLE_PRESETS, CAMERA_MOVEMENTS } from "@/lib/constants";

export const emailSchema = z
  .string()
  .trim()
  .toLowerCase()
  .email("Enter a valid email address")
  .max(254);

export const passwordSchema = z
  .string()
  .min(8, "Password must be at least 8 characters")
  .max(128, "Password must be at most 128 characters")
  .regex(/[a-zA-Z]/, "Password must contain a letter")
  .regex(/[0-9]/, "Password must contain a number");

export const signupSchema = z.object({
  email: emailSchema,
  password: passwordSchema,
  name: z.string().trim().min(1).max(80).optional(),
});

export const loginSchema = z.object({
  email: emailSchema,
  password: z.string().min(1, "Password is required").max(128),
});

export const updateAccountSchema = z
  .object({
    name: z.string().trim().max(80).optional(),
    currentPassword: z.string().max(128).optional(),
    newPassword: passwordSchema.optional(),
  })
  .refine((d) => !d.newPassword || d.currentPassword, {
    message: "Current password is required to set a new password",
    path: ["currentPassword"],
  });

const styleIds = STYLE_PRESETS.map((s) => s.id) as [string, ...string[]];
const cameraIds = CAMERA_MOVEMENTS.map((c) => c.id) as [string, ...string[]];

export const settingsSchema = z.object({
  aspectRatio: z.enum(ASPECT_RATIOS),
  durationSec: z
    .number()
    .int()
    .refine((d) => (DURATIONS as readonly number[]).includes(d), {
      message: `Duration must be one of: ${DURATIONS.join(", ")} seconds`,
    }),
  style: z.enum(styleIds),
  cameraMovement: z.enum(cameraIds),
  motionStrength: z.number().int().min(1).max(10),
  quality: z.enum(QUALITIES),
  seed: z.number().int().min(0).max(2147483647).optional(),
  negativePrompt: z.string().trim().max(1000).optional(),
});

export const promptSchema = z
  .string()
  .trim()
  .min(3, "Describe the video you want in a few words")
  .max(2000, "Prompt must be at most 2000 characters");

export const createGenerationSchema = z.object({
  prompt: promptSchema,
  settings: settingsSchema,
  conversationId: z.string().cuid().optional(),
  /** Client-generated key so retried requests never create duplicate jobs. */
  idempotencyKey: z.string().min(8).max(64).optional(),
  enhance: z.boolean().default(true),
});

export const sendMessageSchema = z.object({
  content: z.string().trim().min(1, "Message cannot be empty").max(4000),
  /** Current draft settings so the assistant can refine them. */
  draft: z
    .object({
      prompt: z.string().max(2000),
      settings: settingsSchema,
    })
    .optional(),
});

export const createConversationSchema = z.object({
  title: z.string().trim().min(1).max(120).optional(),
});

export const listGenerationsSchema = z.object({
  status: z.enum(["QUEUED", "PROCESSING", "COMPLETED", "FAILED"]).optional(),
  search: z.string().trim().max(200).optional(),
  sort: z.enum(["newest", "oldest", "duration", "status"]).default("newest"),
  cursor: z.string().cuid().optional(),
  limit: z.coerce.number().int().min(1).max(50).default(12),
});
