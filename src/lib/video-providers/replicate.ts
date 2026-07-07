import { getEnv } from "@/lib/env";
import {
  CreateJobInput,
  CreateJobResult,
  JobStatus,
  ProviderError,
  VideoProvider,
} from "./provider-interface";

const API_BASE = "https://api.replicate.com/v1";

interface ReplicatePrediction {
  id: string;
  status: "starting" | "processing" | "succeeded" | "failed" | "canceled";
  output?: string | string[] | null;
  error?: string | null;
}

/**
 * Replicate adapter. Works with Replicate-hosted text-to-video models
 * (default: wan-video/wan-2.2-t2v-fast). Model is configurable via
 * REPLICATE_VIDEO_MODEL. Input keys cover the common denominator across
 * Replicate t2v models; unsupported keys are ignored by the API only if the
 * model tolerates them, so we keep the payload minimal.
 */
export class ReplicateProvider implements VideoProvider {
  readonly name = "replicate";

  private headers(): Record<string, string> {
    return {
      Authorization: `Bearer ${getEnv().REPLICATE_API_TOKEN}`,
      "Content-Type": "application/json",
    };
  }

  async createJob(input: CreateJobInput): Promise<CreateJobResult> {
    const env = getEnv();
    const payload: Record<string, unknown> = {
      input: {
        prompt: input.prompt,
        ...(input.negativePrompt ? { negative_prompt: input.negativePrompt } : {}),
        aspect_ratio: input.aspectRatio,
        ...(input.seed !== undefined ? { seed: input.seed } : {}),
      },
      ...(input.webhookUrl
        ? { webhook: input.webhookUrl, webhook_events_filter: ["completed"] }
        : {}),
    };

    const res = await fetch(`${API_BASE}/models/${env.REPLICATE_VIDEO_MODEL}/predictions`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new ProviderError(
        `Replicate request failed (${res.status}): ${body.slice(0, 500)}`,
        this.name
      );
    }
    const prediction = (await res.json()) as ReplicatePrediction;
    return { providerJobId: prediction.id, requestPayload: payload };
  }

  async getJobStatus(providerJobId: string): Promise<JobStatus> {
    const res = await fetch(`${API_BASE}/predictions/${providerJobId}`, {
      headers: this.headers(),
      cache: "no-store",
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new ProviderError(
        `Replicate status check failed (${res.status}): ${body.slice(0, 500)}`,
        this.name
      );
    }
    const prediction = (await res.json()) as ReplicatePrediction;
    switch (prediction.status) {
      case "starting":
        return { state: "queued" };
      case "processing":
        return { state: "processing" };
      case "succeeded": {
        const output = prediction.output;
        const videoUrl = Array.isArray(output) ? output[0] : output ?? undefined;
        if (!videoUrl) {
          return {
            state: "failed",
            errorMessage: "Provider reported success but returned no video URL.",
          };
        }
        return { state: "completed", videoUrl };
      }
      case "canceled":
        return { state: "failed", errorMessage: "Generation was canceled at the provider." };
      case "failed":
      default:
        return {
          state: "failed",
          errorMessage: prediction.error || "The provider failed to generate this video.",
        };
    }
  }
}
