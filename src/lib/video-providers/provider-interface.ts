import type { AspectRatio, Quality } from "@/lib/constants";

export interface CreateJobInput {
  prompt: string;
  negativePrompt?: string;
  aspectRatio: AspectRatio;
  durationSec: number;
  motionStrength: number;
  quality: Quality;
  seed?: number;
  /** Publicly reachable webhook URL, when the deployment supports one. */
  webhookUrl?: string;
}

export interface CreateJobResult {
  providerJobId: string;
  /** Exact payload sent to the provider, stored for debugging/auditing. */
  requestPayload: Record<string, unknown>;
}

export type ProviderJobState = "queued" | "processing" | "completed" | "failed";

export interface JobStatus {
  state: ProviderJobState;
  videoUrl?: string;
  thumbnailUrl?: string;
  errorMessage?: string;
}

/**
 * Adapter contract every text-to-video provider must implement.
 * Add a new provider by implementing this interface and registering it
 * in `provider-factory.ts`.
 */
export interface VideoProvider {
  readonly name: string;
  createJob(input: CreateJobInput): Promise<CreateJobResult>;
  getJobStatus(providerJobId: string): Promise<JobStatus>;
}

export class ProviderError extends Error {
  constructor(
    message: string,
    public readonly providerName: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = "ProviderError";
  }
}
