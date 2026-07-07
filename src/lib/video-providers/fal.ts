import { getEnv } from "@/lib/env";
import {
  CreateJobInput,
  CreateJobResult,
  JobStatus,
  ProviderError,
  VideoProvider,
} from "./provider-interface";

const QUEUE_BASE = "https://queue.fal.run";

interface FalQueueSubmitResponse {
  request_id: string;
}

interface FalQueueStatusResponse {
  status: "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED";
}

interface FalResultResponse {
  video?: { url?: string };
  thumbnail?: { url?: string };
  detail?: unknown;
}

/**
 * fal.ai adapter using the fal queue API (default model: fal-ai/ltx-video,
 * configurable via FAL_VIDEO_MODEL). fal job IDs are stored as
 * "<model>::<request_id>" because the queue API routes by model path.
 */
export class FalProvider implements VideoProvider {
  readonly name = "fal";

  private headers(): Record<string, string> {
    return {
      Authorization: `Key ${getEnv().FAL_KEY}`,
      "Content-Type": "application/json",
    };
  }

  async createJob(input: CreateJobInput): Promise<CreateJobResult> {
    const env = getEnv();
    const payload: Record<string, unknown> = {
      prompt: input.prompt,
      ...(input.negativePrompt ? { negative_prompt: input.negativePrompt } : {}),
      aspect_ratio: input.aspectRatio,
      ...(input.seed !== undefined ? { seed: input.seed } : {}),
    };

    const url = input.webhookUrl
      ? `${QUEUE_BASE}/${env.FAL_VIDEO_MODEL}?fal_webhook=${encodeURIComponent(input.webhookUrl)}`
      : `${QUEUE_BASE}/${env.FAL_VIDEO_MODEL}`;

    const res = await fetch(url, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new ProviderError(
        `fal.ai request failed (${res.status}): ${body.slice(0, 500)}`,
        this.name
      );
    }
    const data = (await res.json()) as FalQueueSubmitResponse;
    return {
      providerJobId: `${env.FAL_VIDEO_MODEL}::${data.request_id}`,
      requestPayload: payload,
    };
  }

  async getJobStatus(providerJobId: string): Promise<JobStatus> {
    const [model, requestId] = providerJobId.split("::");
    if (!model || !requestId) {
      return { state: "failed", errorMessage: "Malformed provider job reference." };
    }

    const statusRes = await fetch(
      `${QUEUE_BASE}/${model}/requests/${requestId}/status`,
      { headers: this.headers(), cache: "no-store" }
    );
    if (!statusRes.ok) {
      const body = await statusRes.text().catch(() => "");
      throw new ProviderError(
        `fal.ai status check failed (${statusRes.status}): ${body.slice(0, 500)}`,
        this.name
      );
    }
    const status = (await statusRes.json()) as FalQueueStatusResponse;
    if (status.status === "IN_QUEUE") return { state: "queued" };
    if (status.status === "IN_PROGRESS") return { state: "processing" };

    // COMPLETED — fetch the result payload.
    const resultRes = await fetch(`${QUEUE_BASE}/${model}/requests/${requestId}`, {
      headers: this.headers(),
      cache: "no-store",
    });
    if (!resultRes.ok) {
      const body = await resultRes.text().catch(() => "");
      // A completed-but-failed job surfaces its error on the result endpoint.
      return {
        state: "failed",
        errorMessage: `The provider reported an error: ${body.slice(0, 300)}`,
      };
    }
    const result = (await resultRes.json()) as FalResultResponse;
    const videoUrl = result.video?.url;
    if (!videoUrl) {
      return {
        state: "failed",
        errorMessage: "Provider finished but returned no video URL.",
      };
    }
    return {
      state: "completed",
      videoUrl,
      thumbnailUrl: result.thumbnail?.url,
    };
  }
}
