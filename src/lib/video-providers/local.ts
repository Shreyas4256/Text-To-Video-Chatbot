import { getEnv } from "@/lib/env";
import {
  CreateJobInput,
  CreateJobResult,
  JobStatus,
  ProviderError,
  VideoProvider,
} from "./provider-interface";

/**
 * Self-hosted provider — talks to the bundled Python inference worker
 * (see worker/main.py) which runs an open-source text-to-video model on
 * your own hardware. No external AI service or API key is involved.
 *
 * LOCAL_WORKER_URL        Where the app reaches the worker (server-side).
 * LOCAL_WORKER_PUBLIC_URL Where the BROWSER reaches the worker for video
 *                         playback (defaults to LOCAL_WORKER_URL).
 * LOCAL_WORKER_TOKEN      Optional shared secret (worker's WORKER_TOKEN).
 */

/** Frame dimensions per aspect ratio (multiples of 8, sized for small models). */
const DIMENSIONS: Record<CreateJobInput["aspectRatio"], { width: number; height: number }> = {
  "16:9": { width: 448, height: 256 },
  "9:16": { width: 256, height: 448 },
  "1:1": { width: 320, height: 320 },
};

const STEPS_BY_QUALITY: Record<CreateJobInput["quality"], number> = {
  draft: 15,
  standard: 25,
  high: 40,
};

const FPS = 8;
/** Small open models are trained on short clips; cap frames accordingly. */
const MAX_FRAMES = 64;

interface WorkerJobResponse {
  job_id: string;
}

interface WorkerStatusResponse {
  status: "queued" | "processing" | "completed" | "failed";
  error: string | null;
  video_url: string | null;
}

export class LocalProvider implements VideoProvider {
  readonly name = "local";

  private baseUrl(): string {
    return getEnv().LOCAL_WORKER_URL.replace(/\/$/, "");
  }

  private headers(): Record<string, string> {
    const env = getEnv();
    return {
      "Content-Type": "application/json",
      ...(env.LOCAL_WORKER_TOKEN
        ? { Authorization: `Bearer ${env.LOCAL_WORKER_TOKEN}` }
        : {}),
    };
  }

  async createJob(input: CreateJobInput): Promise<CreateJobResult> {
    const dims = DIMENSIONS[input.aspectRatio];
    const payload = {
      prompt: input.prompt,
      negative_prompt: input.negativePrompt ?? null,
      num_frames: Math.min(Math.max(input.durationSec * FPS, 8), MAX_FRAMES),
      fps: FPS,
      width: dims.width,
      height: dims.height,
      steps: STEPS_BY_QUALITY[input.quality],
      seed: input.seed ?? null,
    };

    let res: Response;
    try {
      res = await fetch(`${this.baseUrl()}/jobs`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(payload),
      });
    } catch (error) {
      throw new ProviderError(
        `Local worker is unreachable at ${this.baseUrl()}. Is it running? (see worker/README section)`,
        this.name,
        error
      );
    }
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new ProviderError(
        `Local worker rejected the job (${res.status}): ${body.slice(0, 500)}`,
        this.name
      );
    }
    const data = (await res.json()) as WorkerJobResponse;
    return { providerJobId: data.job_id, requestPayload: payload };
  }

  async getJobStatus(providerJobId: string): Promise<JobStatus> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl()}/jobs/${providerJobId}`, {
        headers: this.headers(),
        cache: "no-store",
      });
    } catch (error) {
      throw new ProviderError(
        `Local worker is unreachable at ${this.baseUrl()}.`,
        this.name,
        error
      );
    }
    if (res.status === 404) {
      // Worker restarted and lost its in-memory job — terminal failure.
      return {
        state: "failed",
        errorMessage:
          "The local worker no longer knows this job (it may have restarted). Retry the generation.",
      };
    }
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new ProviderError(
        `Local worker status check failed (${res.status}): ${body.slice(0, 500)}`,
        this.name
      );
    }
    const data = (await res.json()) as WorkerStatusResponse;
    switch (data.status) {
      case "queued":
        return { state: "queued" };
      case "processing":
        return { state: "processing" };
      case "completed": {
        if (!data.video_url) {
          return {
            state: "failed",
            errorMessage: "Worker finished but returned no video file.",
          };
        }
        const env = getEnv();
        const publicBase = (env.LOCAL_WORKER_PUBLIC_URL ?? env.LOCAL_WORKER_URL).replace(/\/$/, "");
        return { state: "completed", videoUrl: `${publicBase}${data.video_url}` };
      }
      case "failed":
      default:
        return {
          state: "failed",
          errorMessage: data.error
            ? `Local model error: ${data.error}`
            : "The local model failed to generate this video.",
        };
    }
  }
}
