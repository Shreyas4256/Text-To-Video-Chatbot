import { z } from "zod";

/**
 * Server-side environment validation. Imported only from server code —
 * never expose these values to the client.
 */
const envSchema = z
  .object({
    NODE_ENV: z.enum(["development", "test", "production"]).default("development"),
    DATABASE_URL: z.string().url({ message: "DATABASE_URL must be a valid connection URL" }),
    SESSION_SECRET: z
      .string()
      .min(32, "SESSION_SECRET must be at least 32 characters"),
    VIDEO_PROVIDER: z.enum(["mock", "replicate", "fal"]),
    REPLICATE_API_TOKEN: z.string().optional(),
    REPLICATE_VIDEO_MODEL: z.string().default("wan-video/wan-2.2-t2v-fast"),
    FAL_KEY: z.string().optional(),
    FAL_VIDEO_MODEL: z.string().default("fal-ai/ltx-video"),
    ANTHROPIC_API_KEY: z.string().optional(),
    ANTHROPIC_MODEL: z.string().default("claude-haiku-4-5"),
    WEBHOOK_SECRET: z.string().optional(),
    APP_URL: z.string().url().optional(),
  })
  .superRefine((env, ctx) => {
    if (env.VIDEO_PROVIDER === "mock" && env.NODE_ENV === "production") {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["VIDEO_PROVIDER"],
        message:
          "VIDEO_PROVIDER=mock is a development-only mode. Configure a real provider (replicate or fal) for production.",
      });
    }
    if (env.VIDEO_PROVIDER === "replicate" && !env.REPLICATE_API_TOKEN) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["REPLICATE_API_TOKEN"],
        message: "REPLICATE_API_TOKEN is required when VIDEO_PROVIDER=replicate",
      });
    }
    if (env.VIDEO_PROVIDER === "fal" && !env.FAL_KEY) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["FAL_KEY"],
        message: "FAL_KEY is required when VIDEO_PROVIDER=fal",
      });
    }
  });

export type Env = z.infer<typeof envSchema>;

let cached: Env | null = null;

/** Validate lazily so `next build` succeeds without runtime secrets. */
export function getEnv(): Env {
  if (cached) return cached;
  const parsed = envSchema.safeParse(process.env);
  if (!parsed.success) {
    const details = parsed.error.issues
      .map((i) => `  - ${i.path.join(".")}: ${i.message}`)
      .join("\n");
    throw new Error(`Invalid environment configuration:\n${details}`);
  }
  cached = parsed.data;
  return cached;
}

export function isMockProvider(): boolean {
  return getEnv().VIDEO_PROVIDER === "mock";
}
