import { getEnv } from "@/lib/env";
import type { VideoProvider } from "./provider-interface";
import { ReplicateProvider } from "./replicate";
import { FalProvider } from "./fal";
import { LocalProvider } from "./local";
import { MockProvider } from "./mock";

let instance: VideoProvider | null = null;

/**
 * Returns the configured provider (VIDEO_PROVIDER env var).
 * Env validation guarantees the required API key is present and that
 * mock mode is never used in production.
 */
export function getVideoProvider(): VideoProvider {
  if (instance) return instance;
  const env = getEnv();
  switch (env.VIDEO_PROVIDER) {
    case "local":
      instance = new LocalProvider();
      break;
    case "replicate":
      instance = new ReplicateProvider();
      break;
    case "fal":
      instance = new FalProvider();
      break;
    case "mock":
      instance = new MockProvider();
      break;
  }
  return instance;
}
